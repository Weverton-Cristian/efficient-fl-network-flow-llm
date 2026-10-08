#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="${PROJECT_ROOT}/data/wifi/raw"
DEST_FILE="${DEST_DIR}/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"
SOURCE_URL="https://cse-cic-ids2018.s3.amazonaws.com/Processed%20Traffic%20Data%20for%20ML%20Algorithms/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"

mkdir -p "${DEST_DIR}"

if command -v wget >/dev/null 2>&1; then
  wget -O "${DEST_FILE}" "${SOURCE_URL}"
elif command -v curl >/dev/null 2>&1; then
  curl -L "${SOURCE_URL}" -o "${DEST_FILE}"
else
  echo "Neither wget nor curl is available. Install one of them to download the dataset." >&2
  exit 1
fi

echo
echo "Dataset saved to:"
echo "  ${DEST_FILE}"
echo
echo "You can now run:"
echo "  python main.py --config configs/config_wifi.yaml"
