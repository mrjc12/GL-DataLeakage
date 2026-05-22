import os

import torch
from time import time
import numpy as np

from core.GNNs.gnn_utils import EarlyStopping
from core.data_utils.load import load_data, load_gpt_preds
from core.utils import time_logger

LOG_FREQ = 10


class GNNTrainer():

    def __init__(self, cfg, feature_type):
        self.seed = cfg.seed
        self.device = cfg.device
        self.dataset_name = cfg.dataset
        self.gnn_model_name = cfg.gnn.model.name
        self.lm_model_name = cfg.lm.model.name
        self.hidden_dim = cfg.gnn.model.hidden_dim
        self.num_layers = cfg.gnn.model.num_layers
        self.dropout = cfg.gnn.train.dropout
        self.lr = cfg.gnn.train.lr
        self.feature_type = feature_type
        self.epochs = cfg.gnn.train.epochs
        self.save_logits = cfg.gnn.train.save_logits

        # ! Load data
        data, num_classes = load_data(
            self.dataset_name, use_dgl=False, use_text=False, seed=self.seed)

        self.num_nodes = data.y.shape[0]
        self.num_classes = num_classes

        # Check Multi-label
        self.multi_label = (data.y.dim() > 1 and data.y.dtype == torch.float32)
        if not self.multi_label:
            data.y = data.y.squeeze().long()

        # ! Init gnn feature
        topk = 3 if self.dataset_name == 'pubmed' else 5
        if self.feature_type == 'ogb':
            print("Loading OGB features...")
            features = data.x
        elif self.feature_type == 'P':
            print("Loading top-k prediction features ...")
            features = load_gpt_preds(self.dataset_name, topk)
        # else:
        #     print(f"Loading pretrained LM features ({self.feature_type}) ...")
        #     LM_emb_path = f"prt_lm/{self.dataset_name}/{self.lm_model_name}_{self.feature_type}-seed{self.seed}.emb"
        #     print(f"LM_emb_path: {LM_emb_path}")
        #
        #     if not os.path.exists(LM_emb_path):
        #         raise FileNotFoundError(
        #             f"Embedding file not found: {LM_emb_path}. Ensure trainLM has been run with text_type={self.feature_type}")
        #
        #     features = torch.from_numpy(np.array(
        #         np.memmap(LM_emb_path, mode='r',
        #                   dtype=np.float16,
        #                   shape=(self.num_nodes, 768)))
        #     ).to(torch.float32)
        else:
            print(f"Loading pretrained LM features ({self.feature_type}) ...")
            LM_emb_path = f"prt_lm/{self.dataset_name}/{self.lm_model_name}_{self.feature_type}-seed{self.seed}.emb"
            print(f"LM_emb_path: {LM_emb_path}")

            if not os.path.exists(LM_emb_path):
                raise FileNotFoundError(
                    f"Embedding file not found: {LM_emb_path}. Ensure trainLM has been run with text_type={self.feature_type}")

            file_size = os.path.getsize(LM_emb_path)
            emb_dim = file_size // (self.num_nodes * 2)

            features = torch.from_numpy(np.array(
                np.memmap(LM_emb_path, mode='r',
                          dtype=np.float16,
                          shape=(self.num_nodes, emb_dim)))
            ).to(torch.float32)

        self.features = features.to(self.device)
        self.data = data.to(self.device)

        # ! Trainer init
        use_pred = self.feature_type == 'P'

        if self.gnn_model_name == "GCN":
            from core.GNNs.GCN.model import GCN as GNN
        elif self.gnn_model_name == "SAGE":
            from core.GNNs.SAGE.model import SAGE as GNN
        elif self.gnn_model_name == "MLP":
            from core.GNNs.MLP.model import MLP as GNN
        else:
            print(f"Model {self.gnn_model_name} is not supported! Loading MLP ...")
            from core.GNNs.MLP.model import MLP as GNN

        self.model = GNN(in_channels=self.hidden_dim * topk if use_pred else self.features.shape[1],
                         hidden_channels=self.hidden_dim,
                         out_channels=self.num_classes,
                         num_layers=self.num_layers,
                         dropout=self.dropout,
                         use_pred=use_pred).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=0.0)

        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)

        print(f"\nNumber of parameters: {trainable_params}")
        self.ckpt = f"output/{self.dataset_name}/{self.gnn_model_name}.pt"
        self.stopper = EarlyStopping(
            patience=cfg.gnn.train.early_stop, path=self.ckpt) if cfg.gnn.train.early_stop > 0 else None

        # Loss function based on label type
        if self.multi_label:
            self.loss_func = torch.nn.BCEWithLogitsLoss()
        else:
            self.loss_func = torch.nn.CrossEntropyLoss()

        from core.GNNs.gnn_utils import Evaluator
        self._evaluator = Evaluator(name=self.dataset_name)
        if self.multi_label:
            self.evaluator = lambda pred, labels: self._evaluator.eval(
                {"y_pred": pred, "y_true": labels})["micro_f1"]
        else:
            self.evaluator = lambda pred, labels: self._evaluator.eval(
                {"y_pred": pred.argmax(dim=-1, keepdim=True),
                 "y_true": labels.view(-1, 1)})["acc"]

    def _forward(self, x, edge_index):
        logits = self.model(x, edge_index)  # small-graph
        return logits

    def _train(self):
        # ! Shared
        self.model.train()
        self.optimizer.zero_grad()
        # ! Specific
        logits = self._forward(self.features, self.data.edge_index)
        loss = self.loss_func(
            logits[self.data.train_mask], self.data.y[self.data.train_mask])
        train_acc = self.evaluator(
            logits[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        self.optimizer.step()

        return loss.item(), train_acc

    @torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        logits = self._forward(self.features, self.data.edge_index)
        if self.multi_label:
            val_acc = self._evaluator.eval(
                {"y_pred": logits[self.data.val_mask], "y_true": self.data.y[self.data.val_mask]})
            test_acc = self._evaluator.eval(
                {"y_pred": logits[self.data.test_mask], "y_true": self.data.y[self.data.test_mask]})
        else:
            val_acc = self.evaluator(
                logits[self.data.val_mask], self.data.y[self.data.val_mask])
            test_acc = self.evaluator(
                logits[self.data.test_mask], self.data.y[self.data.test_mask])
        return val_acc, test_acc, logits

    @time_logger
    def train(self):
        # ! Training
        for epoch in range(self.epochs):
            t0, es_str = time(), ''
            loss, train_acc = self._train()
            val_acc, test_acc, _ = self._evaluate()

            if self.multi_label:
                val_score = val_acc['micro_f1']
            else:
                val_score = val_acc['acc'] if isinstance(val_acc, dict) else val_acc

            if self.stopper is not None:
                es_flag, es_str = self.stopper.step(val_score, self.model, epoch)
                if es_flag:
                    print(
                        f'Early stopped, loading model from epoch-{self.stopper.best_epoch}')
                    break
            if epoch % LOG_FREQ == 0:
                if self.multi_label:
                    print(
                        f'Epoch: {epoch}, Time: {time() - t0:.4f}, Loss: {loss:.4f}, ValMicro: {val_acc["micro_f1"]:.4f}, ValMacro: {val_acc["macro_f1"]:.4f}, ES: {es_str}')
                else:
                    print(
                        f'Epoch: {epoch}, Time: {time() - t0:.4f}, Loss: {loss:.4f}, TrainAcc: {train_acc:.4f}, ValAcc: {val_acc:.4f}, ES: {es_str}')

        # ! Finished training, load checkpoints
        if self.stopper is not None:
            self.model.load_state_dict(torch.load(self.stopper.path))

        return self.model

    @torch.no_grad()
    def eval_and_save(self):
        torch.save(self.model.state_dict(), self.ckpt)
        val_acc, test_acc, logits = self._evaluate()
        if self.save_logits:
            logits_path = f"output/{self.dataset_name}/{self.gnn_model_name}_{self.feature_type}_{self.lm_model_name}-seed{self.seed}.pt"
            os.makedirs(os.path.dirname(logits_path), exist_ok=True)
            torch.save(logits.detach().cpu(), logits_path)
        if self.multi_label:
            print(
                f'[{self.gnn_model_name} + {self.feature_type}] ValMicro: {val_acc["micro_f1"]:.4f}, ValMacro: {val_acc["macro_f1"]:.4f}, TestMicro: {test_acc["micro_f1"]:.4f}, TestMacro: {test_acc["macro_f1"]:.4f}\n')
            res = {'val_acc': val_acc['micro_f1'], 'test_acc': test_acc['micro_f1'], 'val_micro': val_acc['micro_f1'], 'val_macro': val_acc['macro_f1'], 'test_micro': test_acc['micro_f1'], 'test_macro': test_acc['macro_f1']}
        else:
            print(
                f'[{self.gnn_model_name} + {self.feature_type}] Val: {val_acc:.4f}, Test: {test_acc:.4f}\n')
            res = {'val_acc': val_acc, 'test_acc': test_acc}
        return logits, res