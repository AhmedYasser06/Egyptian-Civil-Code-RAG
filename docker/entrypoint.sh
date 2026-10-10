#!/bin/sh
set -e
python -m src.scripts.ensure_index
exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers "${WEB_CONCURRENCY:-1}"
