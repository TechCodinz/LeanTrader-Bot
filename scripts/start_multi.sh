#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="${REPO_DIR}/configs/exchanges.yml"

if ! command -v yq >/dev/null 2>&1; then
  apt-get update -y && apt-get install -y jq curl >/dev/null 2>&1 || true
  curl -sL https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -o /usr/local/bin/yq
  chmod +x /usr/local/bin/yq
fi

mapfile -t EXS < <(yq -r '.exchanges | to_entries[] | select(.value.enabled==true) | .key' "$CFG")

for name in "${EXS[@]}"; do
  env_file=$(yq -r ".exchanges[\"$name\"].env_file" "$CFG")
  mode=$(yq -r ".exchanges[\"$name\"].mode" "$CFG")
  [[ -z "$env_file" || "$env_file" == "null" ]] && continue

  echo "⏳ Spawning $name ($mode)"
  if [[ "$mode" == "live" ]]; then
    echo "  ↳ live handled by router; skipping"
    continue
  fi

  sess="lt-$name"
  tmux has-session -t "$sess" 2>/dev/null && tmux kill-session -t "$sess" || true
  tmux new -d -s "$sess" "set -a; source \"$env_file\"; set +a; cd \"$REPO_DIR\"; \"$REPO_DIR/venv/bin/python\" ultra_launcher.py --mode paper"
  echo "  ✔ started tmux session: $sess"
done

echo "All training profiles launched."


