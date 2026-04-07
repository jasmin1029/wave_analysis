#!/bin/bash
export PATH="$HOME/.local/bin:/opt/render/project/src/.venv/bin:/opt/render/project/python/bin:$PATH"
exec gunicorn app:app --bind "0.0.0.0:${PORT:-10000}" --workers 2 --timeout 120
