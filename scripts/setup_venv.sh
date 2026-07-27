#!/usr/bin/env bash
set -euo pipefail

# Create and activate a virtualenv then install requirements
# Usage: bash scripts/setup_venv.sh

PYTHON=${PYTHON:-python}

${PYTHON} -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Virtualenv created and requirements installed. Activate with: source .venv/bin/activate"
