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
| `social_ready(platform, account?)` | **Pre-flight:** verify the automation browser session is up and logged in | none |
| `social_status` | Show remaining daily budget per platform **and account** | none |
| `social_read_feed(platform, count, account?, url?)` | Read recent posts — **posts are numbered ("1. ", "2. ", …) and each carries its `POST URL:`**. `url` optional: read a specific page's posts (e.g. `url="https://www.linkedin.com/company/<name>/posts"`) instead of the home feed | none (session cap applies) |
| `social_like(platform, url?, account?, post_index?)` | Like a post — `url` optional; `post_index` (default 1) targets the Nth post on the page currently open | 1 interaction |
| `social_comment(platform, text, url?, account?, post_index?)` | Comment on a post — **`url` is OPTIONAL**: pass a `POST URL:` from the feed if you have it, otherwise `post_index` (default 1) targets the Nth post on the page currently open. The tool opens THAT post's comment box and types there. | 1 interaction |
| `social_post(platform, text, account?)` | Publish a text post | 1 post |
| `browser_open / browser_extract / browser_scroll / browser_screenshot` | General reading/research only | none |

Platforms: `x`, `linkedin`. `youtube` and `instagram` are NOT handled by
browser-mcp — use their API-based MCP servers instead.

## Accounts

The `account` parameter selects which profile (and which social identity)
an action runs as. Always specify the account explicitly — do NOT rely on the default (which is `"default"`, an unknown identity). Use `"prateek"` or `"swastik"`.

### Account Mapping

| Account Value | Identity | Brave Profile | Port | MCP Server (LinkedIn native tools) |
|---------------|----------|---------------|------|-----------------------------------|
| `"prateek"` | **Prateek Gupta** — Marketing at Verit Analytics | `~/.brave-profiles/prateek` | 9222 | `linkedin-prateek` |
| `"swastik"` | **Swastik Mishra** — Supply Chain Analytics at Stryker | `~/.brave-profiles/prateek2` | 9223 | `linkedin-swastik` |

Budgets are per account: each account gets 1 post, 10 interactions, and
20 minutes per day. They never share or steal each other's budget.
The agent assigned to one account MUST pass its own account name on every
call and never the other's.

### Agent → Account Assignment

- **`social-media-prateek`** agent → always use `account="prateek"`
- **`social-media-swastik`** agent → always use `account="swastik"`

## Non-negotiable rules

0. **Pre-flight the browser FIRST.** Every turn that will use `social_*` tools
   must start with `social_ready(platform, account)`. It verifies the
   automation browser is up, the profile is logged in, and no login wall /
   challenge is present — read-only, costs no budget. If it returns anything
   other than `READY`, stop and ask the user to fix the browser session.
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
5. **Read before you act.** Fetch the feed with `social_read_feed` — each
   post carries its `POST URL:` permalink. Pass that URL when you have it.
6. **You do NOT need a post URL to like or comment.** If you don't have the
   URL, simply OMIT it and use `post_index` instead: the tool acts on the
   Nth post on the page currently open — this works on the home feed AND on
   company-page post feeds. `social_read_feed` numbers the posts ("1. ",
   "2. ", …) so you can pick the right one — e.g. to comment on the second
   post call `social_comment(platform, text, post_index=2, account=…)`. To
   target a company page's posts, read them first with
   `social_read_feed(platform, count, account=…, url="https://www.linkedin.com/company/<name>/posts")`
   (or `browser_open` the page), then use the number. The tool clicks THAT
   post's Comment button first, so the text always lands in the correct post.
   Never say "I don't have the post URL" — that is not a blocker.
7. **Posts need approval.** Before `social_post`, show the user the exact
   text and get explicit approval. Never publish unapproved content.

## Workflow

1. `social_ready(platform, account)` — browser pre-flight (must be `READY`).
2. `social_status` — check budget.
3. `social_read_feed(platform, count)` or `browser_open(url)` + `browser_extract` — see the content.
4. For likes/comments:
   - If you have the post's `POST URL:` from the feed, pass it.
   - Otherwise use `post_index` — the feed numbers posts ("1. ", "2. ", …):
     `social_comment(platform, text, post_index=2, account=…)` comments on the
     second post of the currently open page. (Open the feed or the page's
     posts first if needed.)
   - Report the tool's result verbatim.
5. For posts: draft text, show the user, get approval, then `social_post`.

## What the tools do automatically (do not duplicate)

- 4–45s humanized gaps, scroll theater, typing rhythm, dwell pauses.
- Daily caps: 1 post / 10 interactions / 20 minutes per platform per day.
- Challenge detection (login walls, captchas, rate limits) → STOP result.
- Post-click verification (e.g. like state confirmed) with no retries.

## Example turn

User: "Like Elon's latest post and comment 'Great take'"
Agent: `social_ready("x", account="prateek")` → `social_status` →
`social_read_feed("x", 5, account="prateek")` (identify the post; use its `POST URL:` if present) →
`social_like("x", "<post-url>", account="prateek")` →
`social_comment("x", "Great take", account="prateek")` (no URL needed — the
feed is open, so it comments on the first post)
→ report results verbatim. If any result starts with STOP/ERROR: stop and
ask the user.
