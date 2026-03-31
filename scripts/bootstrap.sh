#!/usr/bin/env bash
set -euo pipefail

if [ ! -x .venv/Scripts/python.exe ] && [ ! -x .venv/bin/python ]; then
  python -m venv .venv
fi

if [ -x .venv/Scripts/python.exe ]; then
  PY=.venv/Scripts/python.exe
else
  PY=.venv/bin/python
fi

"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r requirements/dev.txt
"$PY" -m pip install -r requirements/api.txt
