#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
"$ROOT/.venv/bin/python" - "$ROOT" <<'PY'
import json,os,signal,sys
from pathlib import Path
root=Path(sys.argv[1]);p=root/'reports/training-state.json'
if not p.exists():print('尚未启动训练');raise SystemExit
s=json.loads(p.read_text());pid=s.get('pid')
try:args=Path(f'/proc/{pid}/cmdline').read_bytes()
except OSError:print('训练进程已结束');raise SystemExit
if str(root/'scripts/train.py').encode() not in args:raise RuntimeError('PID不属于本项目，拒绝停止')
os.kill(pid,signal.SIGINT);print('已请求停止训练；保留最近完整轮次的检查点。')
PY
