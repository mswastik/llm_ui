---
name: social-media-automation
description: >
  Execute social media actions safely on the user's real accounts through the
  browser-mcp server (X and LinkedIn). Use when the user says "post this",
  "like this post", "comment on this", "check my feed", "engage with X",
  "share on LinkedIn", or asks the agent to perform an action on a social
  platform. All pacing, delays, daily limits (1 post, 10 interactions,
  20 minutes per platform per day), and challenge detection are enforced
  DETERMINISTICALLY inside the tools — the agent must never try to add its
  own delays, and must never retry, log in, or solve a captcha when a STOP
  result appears: ask the user to intervene instead. Pairs with the
  social-media-management skill (strategy) — this skill is the execution
  layer only.
version: 1.0.0
---

# Social Media Automation

## Tools (from the `browser-mcp` MCP server)

| Tool | Purpose | Budget |
|---|---|---|
| `social_status` | Show remaining daily budget per platform **and account** | none |
| `social_read_feed(platform, count, account?)` | Read recent posts from a feed (read-only) | none (session cap applies) |
| `social_like(platform, url?, account?)` | Like a post | 1 interaction |
| `social_comment(platform, url, text, account?)` | Comment on a post | 1 interaction |
| `social_post(platform, text, account?)` | Publish a text post | 1 post |
| `browser_open / browser_extract / browser_scroll / browser_screenshot` | General reading/research only | none |

Platforms: `x`, `linkedin`. `youtube` and `instagram` are NOT handled by
browser-mcp — use their API-based MCP servers instead.

## Accounts

The `account` parameter selects which profile (and which social identity)
an action runs as. It is mandatory-optional: every `social_*` and
`browser_*` tool accepts `account` and defaults to `"default"`.

- `"default"` → LinkedIn account A (`~/.brave-profiles/prateek`, port 9222)
- `"a2"` → LinkedIn account B (`~/.brave-profiles/prateek2`, port 9223)

Budgets are per account: account A and account B each get 1 post, 10
interactions, and 20 minutes per day. They never share or steal each
other's budget. The agent assigned to one account MUST pass its own
account name on every call and never the other's.

## Non-negotiable rules

1. **Never retry.** A result starting with `STOP` or `ERROR` means: stop
   immediately, do NOT call the tool again, do NOT try a workaround. Report
   the exact message to the user and ask them to intervene (they may need to
   complete a login/checkpoint in the automation browser window).
2. **Never log in, never solve captchas or checkpoints.** All of that is
   manual, user-only. The automation profile is kept logged in by the user.
3. **Never use `browser_*` tools for social interactions.** Liking,
   commenting, and posting go ONLY through `social_*` tools, which enforce
   the humanized pacing and budget in code. `browser_*` tools are for
   reading pages and research only.
4. **Respect the budget.** `social_status` first. If a platform shows
   EXHAUSTED, do not act on it — tell the user the daily limit is reached
   and stop. Never "batch" actions to work around the limits.
5. **Read before you act.** Fetch the feed or the specific post with
   `social_read_feed` / `browser_extract` first, confirm what you're
   targeting (quote the post), then act.
6. **Posts need approval.** Before `social_post`, show the user the exact
   text and get explicit approval. Never publish unapproved content.

## Workflow

1. `social_status` — check budget.
2. `social_read_feed(platform, count)` or `browser_open(url)` + `browser_extract` — see the content.
3. For likes/comments: act, then report the tool's result verbatim.
4. For posts: draft text, show the user, get approval, then `social_post`.

## What the tools do automatically (do not duplicate)

- 4–45s humanized gaps, scroll theater, typing rhythm, dwell pauses.
- Daily caps: 1 post / 10 interactions / 20 minutes per platform per day.
- Challenge detection (login walls, captchas, rate limits) → STOP result.
- Post-click verification (e.g. like state confirmed) with no retries.

## Example turn

User: "Like Elon's latest post and comment 'Great take'"
Agent: `social_status` → `social_read_feed("x", 5)` (identify the post) →
`social_like("x", "<post-url>")` → `social_comment("x", "<post-url>", "Great take")`
→ report results verbatim. If any result starts with STOP/ERROR: stop and
ask the user.
