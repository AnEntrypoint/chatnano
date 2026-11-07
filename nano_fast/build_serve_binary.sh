#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
py=${CHATNANO_PY:-/data/cache/venvs/ml/bin/python3}
tools=${CHATNANO_PYI_TOOLS:-$HOME/.cache/chatnano/pyinstaller-tools}
out=${CHATNANO_BIN:-$HOME/.cache/chatnano-bin}
work=${CHATNANO_BUILD:-$HOME/.cache/chatnano-build}
if [ ! -d "$tools/PyInstaller" ]; then
  "$py" -m pip install --target "$tools" pyinstaller
fi
PYTHONPATH="$tools" "$py" -m PyInstaller --noconfirm --distpath "$out" --workpath "$work" "$repo/nano_fast/serve.spec"
echo "$out/chatnano-serve/chatnano-serve"
