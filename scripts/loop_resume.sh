#!/bin/zsh
# Headless one-shot resume of the algo-trading task loop.
# Triggered by ~/Library/LaunchAgents/com.shawnteo.algotrading-loop.plist.
# Runs `claude -p` against a self-contained prompt: do ONE task, commit, exit.
# Re-arm by editing the plist's StartCalendarInterval and `launchctl unload && load -w`.

set -uo pipefail

REPO="/Users/shawnteo/documents/github/algo-trading"
cd "$REPO"

# launchd starts with a minimal PATH; restore the user's.
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

TS=$(date +%Y%m%d_%H%M%S)
LOG="scripts/loop_resume_${TS}.log"

PROMPT=$(cat <<'PROMPT_EOF'
You are resuming an autonomous task loop on this local repo (algo-trading).

Steps you MUST perform, in order:

1. cd /Users/shawnteo/documents/github/algo-trading (already CWD).
2. Run `tail -60 LOOP_LOG.md`. The last meaningful entry should be a HANDOVER pointing at the next task.
3. Run `docker compose ps clickhouse`. If clickhouse is not running, run `docker compose up -d clickhouse` and wait until healthy.
4. Read BRIEF.md to find the next task's full text (per the HANDOVER pointer; default order is 7 -> 2 -> 3 -> 4 -> 5 -> 6, Task 1 first).
5. Spawn ONE fresh subagent using the Agent tool with subagent_type "general-purpose". Its prompt MUST include:
   - The full verbatim text of that next task from BRIEF.md
   - The BRIEF's "Working principles" block (verbatim)
   - A "Capacity rule" line telling the subagent to STOP after one sub-deliverable if it senses cap pressure
   - An explicit instruction to commit locally per logical step (no remote push) and to append LOOP_LOG.md + IMPROVEMENTS.md when done
   - A request to report back: commit SHAs, paragraph summary, cuts, IMPROVEMENTS additions
6. Wait for the subagent to finish. Do NOT spawn a second subagent.
7. Append a new HANDOVER entry to LOOP_LOG.md pointing to the next task after this one. Commit it.
8. Print a final message: commit SHAs landed, task name, one-paragraph summary, next HANDOVER target.

Do not push to any git remote (there is none configured).
Do not start a second task in the same run — exit after the one task is done.
If the subagent fails twice in a row, write a blocker entry and stop.
PROMPT_EOF
)

{
  echo "=== loop resume start: ${TS} ==="
  echo "pwd: $(pwd)"
  echo "user: $(whoami)"
  echo "claude: $(which claude) $(claude --version 2>&1 | head -1)"
  echo
  echo "--- git head ---"
  git log --oneline -3
  echo
  echo "--- LOOP_LOG tail ---"
  tail -20 LOOP_LOG.md
  echo
  echo "--- ccusage ---"
  ccusage blocks --json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); a=[b for b in d['blocks'] if b.get('isActive')]; b=a[0] if a else None; print(f'active cost=\${b[\"costUSD\"]:.2f} entries={b[\"entries\"]}') if b else print('no active block (fresh window ready)')" || true
  echo
  echo "--- claude -p (headless one-task run) ---"
  claude -p "${PROMPT}" \
    --dangerously-skip-permissions \
    --max-turns 400 \
    2>&1
  echo
  echo "=== loop resume end: $(date +%Y%m%d_%H%M%S) ==="
} >> "$LOG" 2>&1
