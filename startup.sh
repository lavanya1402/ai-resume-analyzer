#!/usr/bin/env bash
set -euo pipefail

# Run from the directory where the app files live
cd "$(dirname "$0")" || true

# Make sure pip is up-to-date, then install deps (fallback even if Oryx skipped)
python -m pip install --upgrade pip
if [ -f requirements.txt ]; then
  python -m pip install -r requirements.txt
fi

# Launch Streamlit on the expected App Service port (defaults to 8000)
exec python -m streamlit run app.py \
  --server.port="${PORT:-8000}" \
  --server.address=0.0.0.0
