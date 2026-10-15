#!/usr/bin/env bash
# One command for lint + tests + smoke. Exits non-zero on the first failure.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"

echo "== lint";   "$PY" -m ruff check --select F,E9,B .
echo "== tests";  "$PY" -W error -m unittest discover -s tests
echo "== smoke"
"$PY" -m dataset_desk --help >/dev/null
out="$(mktemp -d)"
mkdir -p "$out/proj/outputs/wave-1"
"$PY" - "$out/proj/outputs/wave-1" <<'PYEOF'
import sys
from PIL import Image
for i, c in enumerate([(200, 40, 40), (40, 200, 40), (40, 40, 200)]):
    Image.new("RGB", (32, 32), c).save(f"{sys.argv[1]}/img{i}.png")
PYEOF
"$PY" -m dataset_desk index  --project "$out/proj" --backend none --out "$out/desk" >/dev/null
"$PY" -m dataset_desk report --project "$out/proj" --backend none --out "$out/desk" >/dev/null
echo "verify: ok"
