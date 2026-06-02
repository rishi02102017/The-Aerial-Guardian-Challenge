#!/usr/bin/env bash
# Push current branch using token from .credentials/github_token (gitignored).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOKEN_FILE="$ROOT/.credentials/github_token"
REMOTE="${1:-origin}"
BRANCH="${2:-main}"

if [[ ! -f "$TOKEN_FILE" ]]; then
  echo "Missing $TOKEN_FILE" >&2
  exit 1
fi
TOKEN="$(tr -d '[:space:]' < "$TOKEN_FILE")"
cd "$ROOT"
git push "https://rishi02102017:${TOKEN}@github.com/rishi02102017/The-Aerial-Guardian-Challenge.git" "$BRANCH"
