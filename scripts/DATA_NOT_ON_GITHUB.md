# 大文件说明

以下 arxiv 大文件已随仓库通过 **Git LFS** 提供（`*.pt`、`*.jsonl` 见 `.gitattributes`）：

- `processed_data/arxiv_E.pt`
- `processed_data/arxiv_R.pt`
- `processed_data/arxiv_RM.pt`
- `processed_data/arxiv_fixed_sbert.pt`
- `gpt_response/M/arxiv_M.jsonl`
- `gpt_response/R/arxiv_R.jsonl`

克隆后若文件为指针而非实体，请执行：

```bash
git lfs install
git lfs pull
```

备用下载（Release `data-v1`）：`./scripts/download_release_data.sh`
