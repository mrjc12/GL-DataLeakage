#!/usr/bin/env bash
# 用法: GITHUB_TOKEN=ghp_xxx ./scripts/push_to_github.sh Lucas
set -euo pipefail

GITHUB_USER="${1:-lucas}"
REPO_NAME="${2:-DataLeakage}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "错误: 请先设置 GITHUB_TOKEN"
  echo "  1. 打开 https://github.com/settings/tokens/new"
  echo "  2. 勾选 repo（含私有仓库读写）"
  echo "  3. 生成后执行: export GITHUB_TOKEN='你的token'"
  echo "  4. 再运行: ./scripts/push_to_github.sh ${GITHUB_USER}"
  exit 1
fi

if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "错误: 当前目录不是 git 仓库"
  exit 1
fi

echo "==> 检查/创建私有仓库 ${GITHUB_USER}/${REPO_NAME} ..."
STATUS=$(curl -sS -o /dev/null -w "%{http_code}" \
  -H "Authorization: Bearer ${GITHUB_TOKEN}" \
  -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/${GITHUB_USER}/${REPO_NAME}")

if [[ "$STATUS" == "404" ]]; then
  curl -sS -X POST \
    -H "Authorization: Bearer ${GITHUB_TOKEN}" \
    -H "Accept: application/vnd.github+json" \
    https://api.github.com/user/repos \
    -d "{\"name\":\"${REPO_NAME}\",\"private\":true,\"description\":\"Data Leakage on Text-Attributed Graphs (anonymous submission)\"}" \
    | grep -q '"full_name"' && echo "    已创建私有仓库" || { echo "创建仓库失败"; exit 1; }
elif [[ "$STATUS" == "200" ]]; then
  echo "    仓库已存在，将推送更新"
else
  echo "    无法访问仓库 (HTTP ${STATUS})，请检查用户名与 Token 权限"
  exit 1
fi

REMOTE="https://${GITHUB_TOKEN}@github.com/${GITHUB_USER}/${REPO_NAME}.git"
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REMOTE"
else
  git remote add origin "$REMOTE"
fi

echo "==> 推送 main（含 Git LFS，体积较大，请耐心等待）..."
git push -u origin main

# 推送后改为不含 token 的 URL，避免泄露
git remote set-url origin "https://github.com/${GITHUB_USER}/${REPO_NAME}.git"

echo ""
echo "完成: https://github.com/${GITHUB_USER}/${REPO_NAME} (私有)"
echo "下一步: 打开 https://anonymous.4open.science/ 创建匿名镜像（见 scripts/ANONYMOUS_GITHUB.md）"
