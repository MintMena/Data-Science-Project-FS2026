#!/usr/bin/env bash
# Commit the bot's output files and push, retrying if another run pushed first.
set -euo pipefail
message="$1"
branch="${GITHUB_REF_NAME:-$(git rev-parse --abbrev-ref HEAD)}"
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
for path in tracking TRACKER.md leaderboard/results.csv LEADERBOARD.md; do
  if [ -e "$path" ]; then git add -A -- "$path"; fi
done
if git diff --cached --quiet; then
  echo "Nothing changed."
  exit 0
fi
git commit -m "$message"
for attempt in 1 2 3 4 5; do
  if git push origin "HEAD:$branch"; then exit 0; fi
  sleep $((attempt * 3))
  git pull --rebase origin "$branch"
done
echo "Could not push after 5 tries."
exit 1
