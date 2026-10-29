#!/usr/bin/env bash
set -e
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ${1:-} == --watch ]]; then exec watch -n 5 "$ROOT/.venv/bin/python" "$ROOT/scripts/status.py"; fi
exec "$ROOT/.venv/bin/python" "$ROOT/scripts/status.py"
