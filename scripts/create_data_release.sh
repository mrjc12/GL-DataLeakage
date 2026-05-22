#!/usr/bin/env bash
# 用法: GITHUB_TOKEN=ghp_xxx ./scripts/create_data_release.sh [github_user] [repo] [tag]
set -euo pipefail

GITHUB_USER="${1:-mrjc12}"
REPO_NAME="${2:-DataLeakage}"
TAG="${3:-data-v1}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -z "${GITHUB_TOKEN:-}" && -f "${ROOT}/.github_token" ]]; then
  GITHUB_TOKEN=$(tr -d '\n' < "${ROOT}/.github_token")
  export GITHUB_TOKEN
fi
if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "错误: export GITHUB_TOKEN 或写入 ${ROOT}/.github_token"
  exit 1
fi

FILES=(
  "processed_data/arxiv_E.pt"
  "processed_data/arxiv_R.pt"
  "processed_data/arxiv_RM.pt"
  "processed_data/arxiv_fixed_sbert.pt"
  "gpt_response/M/arxiv_M.jsonl"
  "gpt_response/R/arxiv_R.jsonl"
)

for f in "${FILES[@]}"; do
  [[ -f "$f" ]] || { echo "缺少文件: $f"; exit 1; }
done

api() {
  curl -sS -H "Authorization: Bearer ${GITHUB_TOKEN}" \
    -H "Accept: application/vnd.github+json" "$@"
}

echo "==> 创建或获取 Release ${TAG} ..."
REL_TMP=$(mktemp)
HTTP=$(curl -sS -o "$REL_TMP" -w "%{http_code}" \
  -H "Authorization: Bearer ${GITHUB_TOKEN}" \
  -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/${GITHUB_USER}/${REPO_NAME}/releases/tags/${TAG}")

if [[ "$HTTP" == "200" ]]; then
  UPLOAD_URL=$(python3 -c "import json; d=json.load(open('$REL_TMP')); print(d['upload_url'].split('{')[0])")
  RELEASE_ID=$(python3 -c "import json; d=json.load(open('$REL_TMP')); print(d['id'])")
  echo "    已存在 Release id=${RELEASE_ID}"
else
  api -X POST "https://api.github.com/repos/${GITHUB_USER}/${REPO_NAME}/releases" \
    -d "{\"tag_name\":\"${TAG}\",\"name\":\"Large arxiv data\",\"body\":\"Large arxiv files. See scripts/DATA_NOT_ON_GITHUB.md\",\"draft\":false}" \
    > "$REL_TMP"
  if ! python3 -c "import json; d=json.load(open('$REL_TMP')); assert 'upload_url' in d" 2>/dev/null; then
    python3 -c "import json; d=json.load(open('$REL_TMP')); print('API错误:', d.get('message', d))"
    exit 1
  fi
  UPLOAD_URL=$(python3 -c "import json; d=json.load(open('$REL_TMP')); print(d['upload_url'].split('{')[0])")
  RELEASE_ID=$(python3 -c "import json; d=json.load(open('$REL_TMP')); print(d['id'])")
  echo "    已创建 Release id=${RELEASE_ID}"
fi
rm -f "$REL_TMP"

EXISTING=$(api "https://api.github.com/repos/${GITHUB_USER}/${REPO_NAME}/releases/${RELEASE_ID}/assets" \
  | python3 -c "import sys,json; print(' '.join(a['name'] for a in json.load(sys.stdin)))")

for f in "${FILES[@]}"; do
  name=$(basename "$f")
  if [[ " ${EXISTING} " == *" ${name} "* ]]; then
    echo "==> 跳过 ${name}（已上传）"
    continue
  fi
  echo "==> 上传 ${f} ..."
  curl -fS --progress-bar -X POST \
    -H "Authorization: Bearer ${GITHUB_TOKEN}" \
    -H "Content-Type: application/octet-stream" \
    --data-binary @"${f}" \
    "${UPLOAD_URL}?name=${name}" -o /dev/null
  echo "    OK"
done

echo ""
echo "完成: https://github.com/${GITHUB_USER}/${REPO_NAME}/releases/tag/${TAG}"
