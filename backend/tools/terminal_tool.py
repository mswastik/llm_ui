"""
Terminal tool with layered safety for agent command execution.

Safety model (applied in order):
  1. Hard blocklist   — dangerous patterns, never overridable (accident guard).
  2. Directory allowlist — working_dir must resolve inside an allowed root.
  3. Binary allowlist — allowlisted binaries run directly; others require approval.
  4. Interactive approval — the SSE stream pauses, the user approves/denies in the UI.

Every decision (blocked / pending / approved / denied / executed / timeout) is
appended to the audit log (terminal_audit.jsonl).

NOTE: this is an accident-prevention boundary, not an adversarial sandbox.
"""
import asyncio
import json
import os
import re
import shlex
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from settings import settings_manager

TERMINAL_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "run_command",
        "description": (
            "Execute a shell command on the user's machine. Use for file operations, "
            "running scripts, git, curl, python, package installs, etc. Provide the "
            "full command as a single string. Runs in the project directory by "
            "default; pass working_dir (or prefix 'cd <dir> &&') to run elsewhere — "
            "only directories in the allowlist are permitted. Commands with binaries "
            "outside the allowlist pause for user approval before running."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "working_dir": {
                    "type": "string",
                    "description": "Working directory for the command (optional; must be inside an allowed directory)"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default 120, max 600)"
                }
            },
            "required": ["command"]
        }
    }
}

MAX_OUTPUT_CHARS = 100_000
APPROVAL_TIMEOUT_SECONDS = 900  # 15 min for local use — long enough to step away, short enough to not leak forever

# Hard-coded dangerous patterns — merged with settings, never removable.
HARD_BLOCKED_PATTERNS = [
    r"\brm\s+-[a-z]*[rf][a-z]*\s+/(\s|$)",         # rm -rf / (root itself)
    r"\brm\s+-[a-z]*[rf][a-z]*\s+/\s*\*",          # rm -rf /*
    r"\brm\s+-[a-z]*[rf][a-z]*\s+\.(\s|$)",        # rm -rf .
    r"\brm\s+-[a-z]*[rf][a-z]*\s+(\$HOME|~|/home/|/root/)",  # rm -rf home
    r"\bmkfs\.?\w*",                               # mkfs, mkfs.ext4 ...
    r"\bdd\b[^|;]*\bof=/dev/",                     # dd of=/dev/sd*
    r">\s*/dev/sd\w*",                             # > /dev/sd
    r"\b:\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\};",        # fork bomb
    r"\bsudo\b",                                   # sudo
    r"\bchmod\s+(-R\s+)?[-]?[0-7]{3,4}\s+[~]?/?\s*$",   # chmod 777 / (or ~)
    r"\bchown\s+[^ ]+\s+[~]?/?\s*$",               # chown root /
    r"\b(shutdown|reboot|halt|poweroff)\b",
    r"\bcurl\b[^|;]*\|\s*(ba)?sh\b",               # curl | sh
    r"\bwget\b[^|;]*\|\s*(ba)?sh\b",
    r"python[23]?\s+-c\s+['\"][^'\"]*\bos\.system\b",  # python -c os.system
    r"/dev/(sda|sdb|sdc|nvme|mmcblk\d)",           # raw block devices
    r"\b(init|telinit)\s+[0-6]",
    r"\bpasswd\b",
    r"\bdd\b[^|;]*\bif=/dev/zero",
    r"\b(cryptsetup|lvremove|pvremove|vgremove)\b",
    r"\bopenssl\b[^|;]*\benc\b[^|;]*\b-k\b",       # (loose; informational)
    r"\bkillall?\s+-9\s+(-u\s+)?[a-z_]+",          # kill -9 (broad)
]


class CommandBlocked(Exception):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class _ApprovalGate:
    __slots__ = ("event", "decision")

    def __init__(self):
        self.event = asyncio.Event()
        self.decision: Optional[bool] = None


class ApprovalManager:
    """Registry of pending command-approval gates, keyed by approval_key.

    The terminal tool yields a `tool_approval_required` SSE event, then awaits
    its gate. The FastAPI endpoint `POST /api/tools/{request_id}/approve`
    resolves the gate via `decide()`. A gate with no decision after the timeout
    resolves as denied.
    """

    def __init__(self):
        self._gates: Dict[str, _ApprovalGate] = {}
        self._lock = asyncio.Lock()

    async def _register(self, approval_key: str) -> _ApprovalGate:
        async with self._lock:
            gate = self._gates.get(approval_key)
            if gate is None:
                gate = _ApprovalGate()
                self._gates[approval_key] = gate
            return gate

    def decide(self, approval_key: str, approved: bool) -> bool:
        """Resolve a pending approval. Returns False if no gate is pending."""
        gate = self._gates.get(approval_key)
        if gate is None or gate.decision is not None:
            return False
        gate.decision = approved
        gate.event.set()
        return True

    async def prepare(self, approval_key: str) -> _ApprovalGate:
        """Register a gate BEFORE the approval event is yielded to the UI.

        Pre-registration removes the race where a decision arrives before the
        generator has had a chance to await the gate.
        """
        return await self._register(approval_key)

    async def wait_gate(self, gate: _ApprovalGate, approval_key: str,
                        timeout: float = APPROVAL_TIMEOUT_SECONDS) -> bool:
        """Wait for a decision on an already-prepared gate. Timeout = denied."""
        try:
            await asyncio.wait_for(gate.event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            gate.decision = False
        decision = gate.decision is True
        async with self._lock:
            self._gates.pop(approval_key, None)
        return decision

    async def wait(self, approval_key: str, timeout: float = APPROVAL_TIMEOUT_SECONDS) -> bool:
        """Register + wait in one call (used by tests / direct callers)."""
        gate = await self._register(approval_key)
        return await self.wait_gate(gate, approval_key, timeout)


approval_manager = ApprovalManager()


class TerminalTool:
    """Executes commands with layered safety and streaming progress."""

    def _settings(self) -> Dict[str, Any]:
        return settings_manager.get_settings()

    def _allowed_dirs(self) -> List[str]:
        dirs = self._settings().get("terminal_allowed_dirs") or [".", "./uploads", "./outputs"]
        cwd = os.getcwd()
        resolved = []
        for d in dirs:
            p = d if os.path.isabs(d) else os.path.join(cwd, d)
            resolved.append(os.path.realpath(os.path.abspath(p)))
        return resolved

    def _allowed_commands(self) -> set:
        cmds = self._settings().get("terminal_allowed_commands") or []
        return set(cmds)

    def _blocked_patterns(self) -> List[str]:
        extra = self._settings().get("terminal_blocked_patterns") or []
        return list(HARD_BLOCKED_PATTERNS) + [str(p) for p in extra]

    def _require_approval(self) -> bool:
        v = self._settings().get("terminal_require_approval", True)
        if isinstance(v, bool):
            return v
        return str(v).lower() in ("1", "true", "yes")

    def _audit(self, entry: Dict[str, Any]):
        path = self._settings().get("terminal_audit_log") or "./terminal_audit.jsonl"
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            print(f"[TERMINAL] audit write failed: {e}")

    def check(self, command: str, working_dir: Optional[str]) -> Tuple[str, str, bool]:
        """Run safety checks.

        Returns (resolved_working_dir, binary, needs_approval).
        Raises CommandBlocked for hard violations (blocklist or directory).
        """
        original = command

        # Extract a leading `cd <dir> && cmd` / `cd <dir>; cmd` prefix.
        stripped_command = command
        m = re.match(r"^\s*cd\s+([^&;]+?)\s*(?:&&|;)\s*(.+)$", command)
        if m:
            if not working_dir:
                working_dir = m.group(1).strip()
            stripped_command = m.group(2).strip()

        # 1. Hard blocklist (checked against both original and stripped command).
        for pat in self._blocked_patterns():
            if re.search(pat, original, re.IGNORECASE) or re.search(pat, stripped_command, re.IGNORECASE):
                raise CommandBlocked(f"Command matches blocked pattern: {pat}")

        # 2. Directory allowlist.
        target_dir = os.getcwd()
        if working_dir:
            abs_dir = os.path.abspath(os.path.join(os.getcwd(), working_dir))
            real_dir = os.path.realpath(abs_dir)
            allowed = self._allowed_dirs()
            if not any(real_dir == a or real_dir.startswith(a + os.sep) for a in allowed):
                raise CommandBlocked(f"Working directory not allowed: {working_dir}")
            target_dir = real_dir

        # 3. Binary allowlist.
        try:
            tokens = shlex.split(stripped_command)
        except ValueError:
            tokens = stripped_command.split()
        binary = tokens[0] if tokens else ""
        base = os.path.basename(binary) or binary
        needs_approval = base not in self._allowed_commands()
        return target_dir, base, needs_approval

    async def execute(self, arguments: Dict[str, Any], request_id: str,
                      call_key: Optional[str] = None) -> AsyncGenerator[Dict, None]:
        command = str(arguments.get("command", "")).strip()
        working_dir = arguments.get("working_dir") or None
        default_timeout = self._settings().get("terminal_default_timeout") or 120
        try:
            timeout = int(arguments.get("timeout") or default_timeout)
        except (TypeError, ValueError):
            timeout = int(default_timeout)
        timeout = max(1, min(timeout, 600))
        approval_key = f"{request_id}:{call_key or '0'}"
        audit_base = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "request_id": request_id,
            "command": command,
            "working_dir": working_dir,
            "tool": "run_command",
        }

        if not command:
            yield {"type": "tool_error", "tool": "run_command", "error": "Empty command"}
            return

        try:
            target_dir, binary, needs_approval = self.check(command, working_dir)
        except CommandBlocked as e:
            self._audit({**audit_base, "verdict": "blocked", "reason": e.reason})
            yield {"type": "tool_error", "tool": "run_command",
                   "error": f"Command blocked: {e.reason}"}
            return

        # 4. Interactive approval.
        if needs_approval and self._require_approval():
            self._audit({**audit_base, "verdict": "pending", "binary": binary})
            gate = await approval_manager.prepare(approval_key)
            yield {
                "type": "tool_approval_required",
                "tool": "run_command",
                "command": command,
                "working_dir": target_dir,
                "binary": binary,
                "reason": f"'{binary}' is not in the allowed-commands list — approve to run.",
                "approval_key": approval_key,
            }
            approved = await approval_manager.wait_gate(gate, approval_key)
            is_timeout = not gate.event.is_set()
            self._audit({**audit_base, "verdict": "timeout" if is_timeout else ("approved" if approved else "denied"),
                         "binary": binary})
            if not approved:
                if is_timeout:
                    yield {"type": "tool_error", "tool": "run_command",
                           "error": "Approval timed out after 15 minutes (no response) — command was not executed. Send a follow-up message to retry or approve again."}
                else:
                    yield {"type": "tool_error", "tool": "run_command",
                           "error": "Command denied by user"}
                return

        yield {"type": "tool_progress", "tool": "run_command",
               "status": f"Running: {command}", "progress": 10}
        started = time.monotonic()
        proc = None
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=target_dir,
                env=os.environ.copy(),
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                self._audit({**audit_base, "verdict": "timeout", "binary": binary,
                             "timeout": timeout})
                yield {"type": "tool_error", "tool": "run_command",
                       "error": f"Command timed out after {timeout}s (process killed)"}
                return

            stdout = stdout_b.decode("utf-8", errors="replace")
            stderr = stderr_b.decode("utf-8", errors="replace")
            truncated = False
            if len(stdout) > MAX_OUTPUT_CHARS:
                stdout = stdout[:MAX_OUTPUT_CHARS]
                truncated = True
            if len(stderr) > MAX_OUTPUT_CHARS:
                stderr = stderr[:MAX_OUTPUT_CHARS]
                truncated = True
            duration_ms = int((time.monotonic() - started) * 1000)
            result = {
                "exit_code": proc.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "truncated": truncated,
                "command": command,
                "working_dir": target_dir,
                "duration_ms": duration_ms,
            }
            self._audit({**audit_base, "verdict": "executed", "binary": binary,
                         "exit_code": proc.returncode, "duration_ms": duration_ms})
            yield {
                "type": "tool_progress",
                "tool": "run_command",
                "status": f"Completed (exit {proc.returncode}) in {duration_ms}ms",
                "progress": 100,
                "result": result,
            }
        except asyncio.CancelledError:
            # Client disconnected — kill the child so nothing keeps running.
            if proc is not None and proc.returncode is None:
                proc.kill()
            self._audit({**audit_base, "verdict": "cancelled", "binary": binary})
            raise
        except Exception as e:
            self._audit({**audit_base, "verdict": "error", "binary": binary, "error": str(e)})
            yield {"type": "tool_error", "tool": "run_command", "error": str(e)}
