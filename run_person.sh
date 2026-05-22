#!/bin/bash
types=(NS E NS_E)
base_types=(NS E)
datasets=(ShuiHu HongLou Pantheon)

eval "$(conda shell.bash hook)"
conda activate gen_emb

for dataset in "${datasets[@]}"; do
  for seed in 0 1 2 3; do
    python -m gen_emb.generate --dataset $dataset --text_types "${base_types[@]}" --model e5 --seed $seed --device 0
    python -m gen_emb.generate --dataset $dataset --text_types "${base_types[@]}" --model qwen --seed $seed --device 0
  done
done

conda activate DataLeakage
for dataset in "${datasets[@]}"; do
  out_dir="out_${dataset}"
  mkdir -p "$out_dir"
  for emb_model in "intfloat/multilingual-e5-large" "Qwen/Qwen3-Embedding-0.6B"; do
    [ "$emb_model" = "Qwen/Qwen3-Embedding-0.6B" ] && log_prefix=Qwen || log_prefix=E5
    for type in "${types[@]}"; do
      for model in MLP GCN SAGE; do
        python -m core.trainEnsemble dataset $dataset gnn.train.feature_type $type gnn.model.name $model lm.model.name "$emb_model" >> ${out_dir}/${log_prefix}_${type}_${model}.out
      done
      python -m core.trainEnsemble dataset $dataset gnn.train.feature_type $type gnn.model.name RevGAT lm.model.name "$emb_model" gnn.train.lr 0.002 gnn.train.dropout 0.5 >> ${out_dir}/${log_prefix}_${type}_RevGAT.out
    done
  done
done
