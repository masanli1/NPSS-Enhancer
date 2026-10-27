#!/bin/bash
cd "$(dirname "$0")"
PY="${PYTHON:-/Users/ma/.dsh/dsh-runtimes/dsh-primary-runtime/dependencies/python/bin/python3}"
if [ ! -x "$PY" ]; then
  PY="$(command -v python3)"
fi
exec "$PY" npss_gui.py
