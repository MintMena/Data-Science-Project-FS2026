#!/usr/bin/env bash
# Rebuild the bot's output files on top of the newest version of the repo, commit and push.
# Usage: bash grader/commit.sh "commit message" command to rebuild the files...
# If another run pushed first, it starts again from the newest version, so runs never conflict.
set -euo pipefail
message="$1"
shift
branch="${GITHUB_REF_NAME:-$(git rev-parse --abbrev-ref HEAD)}"
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
for attempt in 1 2 3 4 5; do
  git fetch -q origin "$branch"
  git reset -q --hard "origin/$branch"
  "$@"
  for path in tracking TRACKER.md leaderboard/results.csv LEADERBOARD.md; do
    if [ -e "$path" ]; then git add -A -- "$path"; fi
  done
  if git diff --cached --quiet; then
    echo "Nothing changed."
    exit 0
  fi
  git commit -q -m "$message"
  if git push -q origin "HEAD:$branch"; then
    echo "Pushed on try $attempt."
    exit 0
  fi
  echo "Someone else pushed first. Trying again."
  sleep $((attempt * 3))
done
echo "Could not push after 5 tries."
exit 1
