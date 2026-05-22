# 未上传到 GitHub 的大文件

以下文件因超过 GitHub 单文件 100MB 限制或 LFS 免费额度，仅保留在本地：

| 文件 | 约大小 | 本地生成方式 |
|------|--------|--------------|
| `processed_data/arxiv_E.pt` | 128MB | `generate_R_E.py` / 数据处理流程 |
| `processed_data/arxiv_R.pt` | 207MB | 同上 |
| `processed_data/arxiv_RM.pt` | 287MB | 同上 |
| `processed_data/arxiv_fixed_sbert.pt` | 668MB | 嵌入生成 |
| `gpt_response/M/arxiv_M.jsonl` | 438MB | `generate_M.py` |
| `gpt_response/R/arxiv_R.jsonl` | 620MB | `generate_R_E.py` |

其余 `processed_data/*.pt` 与 `gpt_response/` 中小文件已随仓库上传（Git LFS）。

若需完整备份，请使用本地磁盘、机构网盘或 GitHub LFS 付费包。
