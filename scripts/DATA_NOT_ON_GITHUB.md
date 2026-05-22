# 大文件（GitHub Release）

以下文件超过 Git 单文件 100MB，未纳入仓库 / LFS，请从 **Release `data-v1`** 下载：

https://github.com/mrjc12/DataLeakage/releases/tag/data-v1

| Release 资源名 | 仓库内路径 | 约大小 |
|----------------|------------|--------|
| `arxiv_E.pt` | `processed_data/arxiv_E.pt` | 128MB |
| `arxiv_R.pt` | `processed_data/arxiv_R.pt` | 207MB |
| `arxiv_RM.pt` | `processed_data/arxiv_RM.pt` | 287MB |
| `arxiv_fixed_sbert.pt` | `processed_data/arxiv_fixed_sbert.pt` | 668MB |
| `arxiv_M.jsonl` | `gpt_response/M/arxiv_M.jsonl` | 438MB |
| `arxiv_R.jsonl` | `gpt_response/R/arxiv_R.jsonl` | 620MB |

下载示例（在仓库根目录执行）：

```bash
TAG=data-v1
BASE=https://github.com/mrjc12/DataLeakage/releases/download/${TAG}
mkdir -p processed_data gpt_response/M gpt_response/R
curl -L -o processed_data/arxiv_E.pt            "${BASE}/arxiv_E.pt"
curl -L -o processed_data/arxiv_R.pt            "${BASE}/arxiv_R.pt"
curl -L -o processed_data/arxiv_RM.pt           "${BASE}/arxiv_RM.pt"
curl -L -o processed_data/arxiv_fixed_sbert.pt  "${BASE}/arxiv_fixed_sbert.pt"
curl -L -o gpt_response/M/arxiv_M.jsonl           "${BASE}/arxiv_M.jsonl"
curl -L -o gpt_response/R/arxiv_R.jsonl           "${BASE}/arxiv_R.jsonl"
```

维护者上传 / 更新 Release：

```bash
export GITHUB_TOKEN='你的token'
chmod +x scripts/create_data_release.sh
./scripts/create_data_release.sh
```

其余 `processed_data/*.pt` 与 `gpt_response/` 中小文件已随仓库上传（Git LFS）。
