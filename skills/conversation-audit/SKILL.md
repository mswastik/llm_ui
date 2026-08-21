---
name: conversation-audit
description: Read chat contents from llm_ui.db to identify and compare duplicate conversations by actual depth/quality (not just timestamps).
---

Load this skill when the user wants to audit, deduplicate, or compare llm_ui conversation threads. The app's built-in list_conversations tool only returns metadata (title, agent, timestamps), not message bodies — so this script reads the SQLite DB directly at /home/swastik/Downloads/repos/llm_ui/llm_ui.db.

Run: python3 tools/conversation_audit.py [--top|--duplicates|--dump <conv_id>|--json]
- Default: summary table of all conversations (msg counts, char/token estimates, depth score).
- --top: sort by depth_score (assistant response length + count) so the richest conversation wins.
- --duplicates: group title-similar conversations and flag likely duplicates.
- --dump <conv_id>: print full role/content of every message in one conversation.
- --json: machine-readable output for programmatic comparison.

Use the depth_score (assistant_total_chars / 200 + assistant_msg_count * 15) to decide which duplicate to KEEP — higher = more detailed/higher-quality response. Delete the lower-scoring duplicates after user confirmation. Never delete without explicit user approval.
