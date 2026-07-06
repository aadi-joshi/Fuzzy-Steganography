#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Upload final.ipynb to an active Kaggle Jupyter kernel session.
#
# USAGE:
#   1. Open your Kaggle notebook and copy the session URL (the one that looks like
#      https://kkb-production.jupyter-proxy.kaggle.net?token=eyJ...)
#   2. Run:  bash scripts/upload_to_kaggle.sh "YOUR_FULL_URL_WITH_TOKEN"
#
# The script will:
#   • Extract the token from the URL
#   • Upload final.ipynb to the running kernel
#   • Start execution via the Jupyter REST API
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
NOTEBOOK="${REPO_ROOT}/final.ipynb"

FULL_URL="${1:-}"
if [[ -z "$FULL_URL" ]]; then
    echo "Usage: bash scripts/upload_to_kaggle.sh 'https://kkb-production.jupyter-proxy.kaggle.net?token=eyJ...'"
    exit 1
fi

# Extract base URL and token
BASE_URL="$(echo "$FULL_URL" | cut -d'?' -f1)"
TOKEN="$(echo "$FULL_URL" | sed 's/.*token=//')"

echo "Base URL : $BASE_URL"
echo "Token    : ${TOKEN:0:20}..."

# Upload the notebook
echo ""
echo "Uploading final.ipynb..."
PAYLOAD=$(python3 -c "
import json, sys
nb = json.load(open(r'${NOTEBOOK}'))
body = {'type': 'notebook', 'format': 'json', 'content': nb}
print(json.dumps(body))
")

HTTP=$(curl -s -o /tmp/upload_resp.json -w '%{http_code}' \
    -X PUT \
    -H "Authorization: token $TOKEN" \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD" \
    "${BASE_URL}/api/contents/final.ipynb")

echo "Upload HTTP status: $HTTP"
if [[ "$HTTP" == "200" || "$HTTP" == "201" ]]; then
    echo "✓ Notebook uploaded successfully!"
    echo ""
    echo "Starting a kernel to execute it..."
    KERNEL_RESP=$(curl -s -X POST \
        -H "Authorization: token $TOKEN" \
        -H "Content-Type: application/json" \
        -d '{"name":"python3"}' \
        "${BASE_URL}/api/kernels")
    KERNEL_ID=$(echo "$KERNEL_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('id',''))")
    echo "Kernel ID: $KERNEL_ID"
    echo ""
    echo "Your notebook is ready. Open it in the Kaggle UI and click 'Run All'."
else
    echo "✗ Upload failed (HTTP $HTTP). Is the session still active?"
    cat /tmp/upload_resp.json
fi
