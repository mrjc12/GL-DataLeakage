import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from core.utils import init_random_state


def _default_nodes_edges_paths(root_dir):
    base = os.path.join(root_dir, "dataset", "Pantheon")
    return (
        os.path.join(base, "Nodes.csv"),
        os.path.join(base, "Edges.csv"),
    )


def _compute_degree(node_ids, edges_df):
    degree_count = {nid: 0 for nid in node_ids}
    for _, edge in edges_df.iterrows():
        u, v = str(int(float(edge["Source"]))), str(int(float(edge["Target"])))
        if u in degree_count and v in degree_count and u != v:
            degree_count[u] += 1
            degree_count[v] += 1
    return degree_count


def _sample_15_per_class(bucket_ids_all, nid_to_label, degree_count, sort_by_degree=False):
    sampled = []
    target_labels = [
        "AFRICA", "ASIA", "EUROPE", "NORTH AMERICA",
        "OCEANIA", "SOUTH AMERICA", "UNKNOWN",
    ]
    counts = {label: 0 for label in target_labels}
    for label in target_labels:
        label_nodes = [nid for nid in bucket_ids_all if nid_to_label.get(nid) == label]
        if sort_by_degree:
            label_nodes = sorted(label_nodes, key=lambda x: degree_count[x], reverse=True)
            sampled_nodes = label_nodes[:15]
        else:
            if len(label_nodes) >= 15:
                sampled_nodes = np.random.choice(label_nodes, 15, replace=False).tolist()
            else:
                sampled_nodes = label_nodes
        sampled.extend(sampled_nodes)
        counts[label] = len(sampled_nodes)
    avg_degree = sum(degree_count[nid] for nid in sampled) / len(sampled) if sampled else 0.0
    return set(sampled), counts, avg_degree


def _sample_d1_low_degree(bucket_ids_all, nid_to_label, degree_count, per_class=15):
    target_labels = [
        "AFRICA", "ASIA", "EUROPE", "NORTH AMERICA",
        "OCEANIA", "SOUTH AMERICA", "UNKNOWN",
    ]
    remaining = {label: per_class for label in target_labels}
    counts = {label: 0 for label in target_labels}
    sampled = []

    for nid in sorted(bucket_ids_all, key=lambda x: degree_count[x]):
        label = nid_to_label.get(nid)
        if label not in remaining or remaining[label] <= 0:
            continue
        sampled.append(nid)
        remaining[label] -= 1
        counts[label] += 1

    avg_degree = sum(degree_count[nid] for nid in sampled) / len(sampled) if sampled else 0.0
    return set(sampled), counts, avg_degree


def _sample_d3_high_degree(bucket_ids_all, nid_to_label, degree_count, per_class=15):
    target_labels = [
        "AFRICA", "ASIA", "EUROPE", "NORTH AMERICA",
        "OCEANIA", "SOUTH AMERICA", "UNKNOWN",
    ]
    remaining = {label: per_class for label in target_labels}
    counts = {label: 0 for label in target_labels}
    sampled = []

    for nid in sorted(bucket_ids_all, key=lambda x: degree_count[x], reverse=True):
        label = nid_to_label.get(nid)
        if label not in remaining or remaining[label] <= 0:
            continue
        sampled.append(nid)
        remaining[label] -= 1
        counts[label] += 1

    avg_degree = sum(degree_count[nid] for nid in sampled) / len(sampled) if sampled else 0.0
    return set(sampled), counts, avg_degree


def _build_texts(nodes_df, bucket_ids):
    texts = []
    for _, row in nodes_df.iterrows():
        nid = str(int(float(row["Id"])))
        name = str(row["Name"])
        pid = str(row["PERSON_ID"])
        if nid in bucket_ids:
            des_text = str(row["des"]) if pd.notna(row["des"]) else ""
            texts.append(f"Person:{name}\nIntroduction:{des_text}")
        else:
            n_des_text = str(row["N_des"]) if pd.notna(row["N_des"]) else ""
            texts.append(f"Person:[{pid}]\nIntroduction:{n_des_text}")
    return texts


def _build_leak_signal(nodes_df, bucket_ids):
    flags = []
    for _, row in nodes_df.iterrows():
        nid = str(int(float(row["Id"])))
        flags.append(1.0 if nid in bucket_ids else 0.0)
    return torch.tensor(flags, dtype=torch.float32).view(-1, 1)


def _get_sbert_embeddings(texts, batch_size=64):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return torch.from_numpy(emb.astype(np.float32))


def _random_masks(num_nodes, train_ratio=0.6, val_ratio=0.2):
    perm = torch.randperm(num_nodes)
    train_num = int(num_nodes * train_ratio)
    val_num = int(num_nodes * val_ratio)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[perm[:train_num]] = True
    val_mask[perm[train_num:train_num + val_num]] = True
    test_mask[perm[train_num + val_num:]] = True
    return train_mask, val_mask, test_mask


def build_and_save_bucket(
    nodes_df,
    edges_df,
    node_to_index,
    bucket_name,
    bucket_ids,
    labels,
    label_to_idx,
    out_path,
    split,
    embedding_type,
    reuse_x_from=None,
):
    texts = _build_texts(nodes_df, bucket_ids)
    leak_signal = _build_leak_signal(nodes_df, bucket_ids)

    y, cat_names = [], []
    for _, row in nodes_df.iterrows():
        label_str = str(row["continentName"])
        if label_str not in label_to_idx:
            raise ValueError(f"Invalid label: {label_str}")
        y.append(label_to_idx[label_str])
        cat_names.append(label_str)

    row_idx, col_idx = [], []
    for _, edge in edges_df.iterrows():
        u, v = str(int(float(edge["Source"]))), str(int(float(edge["Target"])))
        if u in node_to_index and v in node_to_index:
            i_u, i_v = node_to_index[u], node_to_index[v]
            if i_u != i_v:
                row_idx.extend([i_u, i_v])
                col_idx.extend([i_v, i_u])
    edge_index = (
        torch.tensor([row_idx, col_idx], dtype=torch.long)
        if row_idx else torch.empty((2, 0), dtype=torch.long)
    )

    if reuse_x_from and os.path.isfile(reuse_x_from):
        old = torch.load(reuse_x_from, map_location="cpu", weights_only=False)
        x = old.x.float() if hasattr(old, "x") else old["x"].float()
    elif embedding_type == "sbert":
        x = _get_sbert_embeddings(texts)
    else:
        raise ValueError(f"Unsupported embedding_type: {embedding_type}")

    data = Data()
    data.x = x
    data.leak_signal = leak_signal
    data.y = torch.tensor(y, dtype=torch.long)
    data.raw_texts = texts
    data.category_names = cat_names
    data.label_names = labels
    data.edge_index = edge_index
    data.num_nodes = leak_signal.size(0)

    n_ones = int(leak_signal.sum().item())
    print(
        f"{bucket_name}: leak_signal 1={n_ones}, 0={data.num_nodes - n_ones}, "
        f"total={data.num_nodes}"
    )

    split_seeds = list(range(10))
    train_masks, val_masks, test_masks = [], [], []
    for k in split_seeds:
        init_random_state(k)
        tr, va, te = _random_masks(data.num_nodes)
        train_masks.append(tr)
        val_masks.append(va)
        test_masks.append(te)
    data.train_masks = train_masks
    data.val_masks = val_masks
    data.test_masks = test_masks

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(data, out_path)
    print(f"saved: {out_path}")
    return data


def Pantheon_D_to_graph_singleLabel(
    nodes_path,
    edges_path,
    out_root="processed_data",
    split="random",
    embedding_type="sbert",
    reuse_x_from_existing=True,
):
    labels = [
        "AFRICA", "ASIA", "EUROPE", "NORTH AMERICA",
        "OCEANIA", "SOUTH AMERICA", "UNKNOWN",
    ]
    label_to_idx = {name: i for i, name in enumerate(labels)}

    nodes_df = pd.read_csv(nodes_path, sep=None, engine="python")
    edges_df = pd.read_csv(edges_path, sep=None, engine="python")

    node_ids = [str(int(float(row["Id"]))) for _, row in nodes_df.iterrows()]
    node_to_index = {nid: idx for idx, nid in enumerate(node_ids)}
    degree_count = _compute_degree(node_ids, edges_df)

    d1_ids_all = [nid for nid, d in degree_count.items() if 1 < d <= 4]
    d2_ids_all = [nid for nid, d in degree_count.items() if 4 < d <= 16]
    d3_ids_all = [nid for nid, d in degree_count.items() if d > 16]

    nid_to_label = {
        str(int(float(row["Id"]))): str(row["continentName"])
        for _, row in nodes_df.iterrows()
    }

    init_random_state(0)
    d1_ids, d1_counts, d1_avg_deg = _sample_d1_low_degree(
        d1_ids_all, nid_to_label, degree_count
    )
    d2_ids, d2_counts, d2_avg_deg = _sample_15_per_class(
        d2_ids_all, nid_to_label, degree_count, sort_by_degree=True
    )
    d3_ids, d3_counts, d3_avg_deg = _sample_d3_high_degree(
        d3_ids_all, nid_to_label, degree_count
    )

    print(f"D1(>1,<=4) sampled={len(d1_ids)}, avg_degree={d1_avg_deg:.2f}")
    for k, v in d1_counts.items():
        print(f"  {k}: {v}")
    print(f"D2(>4,<=16) sampled={len(d2_ids)}, avg_degree={d2_avg_deg:.2f}")
    for k, v in d2_counts.items():
        print(f"  {k}: {v}")
    print(f"D3(>16) sampled={len(d3_ids)}, avg_degree={d3_avg_deg:.2f}")
    for k, v in d3_counts.items():
        print(f"  {k}: {v}")

    buckets = [("D1", d1_ids), ("D2", d2_ids), ("D3", d3_ids)]
    results = {}
    for bucket_name, bucket_ids in buckets:
        out_path = os.path.join(
            out_root, f"Pantheon-{bucket_name}_{split}_{embedding_type}.pt"
        )
        reuse = out_path if reuse_x_from_existing else None
        data = build_and_save_bucket(
            nodes_df,
            edges_df,
            node_to_index,
            bucket_name,
            bucket_ids,
            labels,
            label_to_idx,
            out_path,
            split,
            embedding_type,
            reuse_x_from=reuse,
        )
        results[bucket_name] = data
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    parser.add_argument("--nodes_path", type=str, default="")
    parser.add_argument("--edges_path", type=str, default="")
    parser.add_argument("--out_root", type=str, default="processed_data")
    parser.add_argument("--split", type=str, default="random")
    parser.add_argument("--embedding_type", type=str, default="sbert")
    parser.add_argument("--recompute_x", action="store_true")
    args = parser.parse_args()

    nodes_path, edges_path = _default_nodes_edges_paths(args.root)
    if args.nodes_path:
        nodes_path = args.nodes_path
    if args.edges_path:
        edges_path = args.edges_path

    Pantheon_D_to_graph_singleLabel(
        nodes_path=nodes_path,
        edges_path=edges_path,
        out_root=args.out_root,
        split=args.split,
        embedding_type=args.embedding_type,
        reuse_x_from_existing=not args.recompute_x,
    )

# python -m core.data_utils.build_pantheon_d
