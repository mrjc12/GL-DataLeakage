# core/data_utils/load.py
import os
import sys
import torch
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
from torch_geometric.data import Data
from core.config import cfg


class DGLDatasetWrapper:
    """Wrapper to mimic the interface expected by DGLGNNTrainer"""
    def __init__(self, g, train_mask, val_mask, test_mask):
        self.g = g
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

    def __getitem__(self, idx):
        # DGLTrainer expects dataset[0] to return the graph
        return self.g

    def __len__(self):
        return 1


def load_data(dataset, use_dgl=False, use_text=False, text_type='TA', seed=0):
    # Construct path: processed_data/{dataset}_{split}_{format}.pt
    if dataset == 'arxiv':
        file_path = os.path.join(cfg.data.root, f'{dataset}_fixed_{cfg.data.format}.pt')
    else:
        file_path = os.path.join(cfg.data.root, f'{dataset}_{cfg.data.split}_{cfg.data.format}.pt')

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")

    # Load content
    data_dict = torch.load(file_path)

    # Extract basic graph data
    x = data_dict['x']
    y = data_dict['y']
    edge_index = data_dict['edge_index']

    # Select mask based on seed (cycling if seed exceeds list length)
    if dataset == 'arxiv':
        # Load official split from OGB for arxiv dataset
        # Disable interactive prompt by redirecting stdin to auto-answer 'N'
        
        # ogb_dataset = PygNodePropPredDataset(
        #     name='ogbn-arxiv', transform=T.ToSparseTensor())
        # ogb_data = ogb_dataset[0]

        # split_idx = ogb_dataset.get_idx_split()

        import io
        old_stdin = sys.stdin
        # Create a StringIO that can provide multiple 'N\n' responses
        class NonInteractiveInput(io.StringIO):
            def read(self, size=-1):
                return 'N\n'
            def readline(self, size=-1):
                return 'N\n'
        sys.stdin = NonInteractiveInput()
        try:
            ogb_dataset = PygNodePropPredDataset(
                name='ogbn-arxiv', transform=T.ToSparseTensor())
            ogb_data = ogb_dataset[0]
            split_idx = ogb_dataset.get_idx_split()
        finally:
            sys.stdin = old_stdin
        # Create masks with the same size as the processed data
        num_nodes = x.shape[0]
        train_mask = torch.zeros(num_nodes).bool()
        val_mask = torch.zeros(num_nodes).bool()
        test_mask = torch.zeros(num_nodes).bool()
        
        # Map OGB indices to processed data indices
        # Note: This assumes the node order is the same between OGB and processed data
        # Verify node count matches
        if ogb_data.num_nodes != num_nodes:
            raise ValueError(
                f"Node count mismatch: OGB dataset has {ogb_data.num_nodes} nodes, "
                f"but processed data has {num_nodes} nodes. "
                f"Please ensure the processed data maintains the same node order as OGB dataset."
            )
        
        train_mask[split_idx['train']] = True
        val_mask[split_idx['valid']] = True
        test_mask[split_idx['test']] = True

    else:
        num_splits = len(data_dict['train_masks'])
        split_idx = seed % num_splits

        train_mask = data_dict['train_masks'][split_idx]
        val_mask = data_dict['val_masks'][split_idx]
        test_mask = data_dict['test_masks'][split_idx]
    num_classes = len(data_dict['label_names'])

    # 1. Prepare Data Object (PyG or DGL Wrapper)
    if use_dgl:
        import dgl
        # Convert to DGL Graph
        g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=x.shape[0])

        # Add features and labels
        g.ndata['feat'] = x
        g.ndata['label'] = y
        g.ndata['train_mask'] = train_mask
        g.ndata['val_mask'] = val_mask
        g.ndata['test_mask'] = test_mask

        g = dgl.add_self_loop(g)

        data = DGLDatasetWrapper(g, train_mask, val_mask, test_mask)
    else:
        # Create PyG Data Object
        data = Data(x=x, y=y, edge_index=edge_index)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        data.num_nodes = x.shape[0]

    # 2. Prepare Text Data if requested
    if use_text:
        if text_type == 'TA':
            text = data_dict['raw_texts']
        else:
            # load pt
            fname = 'explanation' if text_type == 'E' else text_type

            feat_path = os.path.join(cfg.data.root, f'{dataset}_{fname}.pt')
            if not os.path.exists(feat_path):
                raise FileNotFoundError(f"Text feature file not found at {feat_path}")
            text = torch.load(feat_path)

        return data, num_classes, text

    return data, num_classes


def load_gpt_preds(dataset_name, topk):
    """Load prediction indices (Already indices, shape [N, topk])"""
    pred_path = os.path.join(cfg.data.root, f'{dataset_name}_pred.pt')

    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Prediction file not found at {pred_path}")

    preds = torch.load(pred_path)

    if preds.size(-1) > topk:
        preds = preds[:, :topk]

    return preds.long()
