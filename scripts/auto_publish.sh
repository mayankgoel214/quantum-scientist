#!/usr/bin/env bash
# Keep the public artifacts in sync with IBM until judging.
# Every few minutes: collect any finished hardware jobs; if new results landed, rebuild the paper
# (with a small Gemini polish allowance), re-export the static site, deploy to Vercel and push to GitHub.
#
#   scripts/auto_publish.sh runs/<run_id> [stop_epoch]
set -u
RUN="${1:?run dir}"
STOP="${2:-$(date -j -f '%Y-%m-%d %H:%M' "$(date +%Y-%m-%d) 09:50" +%s)}"
cd "$(dirname "$0")/.."
count() { ls "$RUN"/experiments/*.json 2>/dev/null | wc -l | tr -d ' '; }
published=$(count)
echo "$(date +%T) watching $RUN, $published results published, stop at $(date -r "$STOP" +%T)"
while [ "$(date +%s)" -lt "$STOP" ]; do
  PYTHONUNBUFFERED=1 uv run python -m scripts.qpu_batch collect "$RUN" 2>&1 | grep -v WARNING | tail -1
  now=$(count)
  if [ "$now" -gt "$published" ]; then
    echo "$(date +%T) $now results (was $published): republishing"
    python3 - "$RUN/budget.json" <<'EOF'
import json, sys
p = sys.argv[1]; d = json.load(open(p))
d["max_llm_calls"] = d["llm_calls"] + 4   # two sections, one retry each
json.dump(d, open(p, "w"))
EOF
    uv run python -m scripts.write_paper "$RUN" 2>&1 | grep -E "PDF|polish|paper\]"
    uv run python -m scripts.export_static >/dev/null 2>&1 && (cd site && vercel deploy --prod --yes >/dev/null 2>&1) \
      && echo "$(date +%T) site deployed" || echo "$(date +%T) DEPLOY FAILED"
    git add "$RUN" && git commit -qm "Hardware results update: $now experiments returned from ibm_fez" \
      && GIT_HTTP_LOW_SPEED_LIMIT=1000 GIT_HTTP_LOW_SPEED_TIME=30 git push -q 2>&1 | tail -1
    published=$now
  fi
  sleep 180
done
echo "$(date +%T) stopped; $published results published"
