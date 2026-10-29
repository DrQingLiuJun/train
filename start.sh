#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"
name=main
args=("$@")
for ((i=0;i<${#args[@]};i++)); do if [[ ${args[i]} == --name ]]; then name=${args[i+1]:-}; fi; done
[[ "$name" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid run name'; exit 2; }
if [[ -f .train.pid ]] && kill -0 "$(cat .train.pid)" 2>/dev/null; then echo '已有训练进程，请先查看 ./status.sh'; exit 1; fi
mkdir -p "runs/$name" reports
nohup setsid flock -n "$ROOT/.train.lock" "$ROOT/.venv/bin/python" -u -B "$ROOT/scripts/train.py" "$@" >>"runs/$name/train.log" 2>&1 < /dev/null &
echo $! > .train.pid
printf '训练启动请求已提交，PID=%s\n日志：%s/runs/%s/train.log\n' "$(cat .train.pid)" "$ROOT" "$name"
