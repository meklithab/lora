#!/usr/bin/env bash
# usage: scripts/dev/commit.sh "2026-08-17T14:20:00" "commit message"
set -euo pipefail
export GIT_AUTHOR_DATE="$1"
export GIT_COMMITTER_DATE="$1"
git commit -m "$2"
