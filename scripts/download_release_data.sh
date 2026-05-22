#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG="${1:-data-v1}"
BASE="https://github.com/mrjc12/DataLeakage/releases/download/${TAG}"
cd "$ROOT"
mkdir -p processed_data gpt_response/M gpt_response/R
download() { local out="$1" name="$2"; echo "==> $out"; curl -fL --retry 3 -C - -o "$out" "${BASE}/${name}"; }
download processed_data/arxiv_E.pt arxiv_E.pt
download processed_data/arxiv_R.pt arxiv_R.pt
download processed_data/arxiv_RM.pt arxiv_RM.pt
download processed_data/arxiv_fixed_sbert.pt arxiv_fixed_sbert.pt
download gpt_response/M/arxiv_M.jsonl arxiv_M.jsonl
download gpt_response/R/arxiv_R.jsonl arxiv_R.jsonl
echo "Done: ${TAG}"
