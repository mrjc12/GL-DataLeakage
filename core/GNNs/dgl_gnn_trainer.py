import torch
from time import time
import numpy as np
import os

from core.GNNs.RevGAT.model import RevGAT
from core.GNNs.gnn_utils import EarlyStopping
from core.data_utils.load import load_data, load_gpt_preds
from core.utils import time_logger

LOG_FREQ = 10


class DGLGNNTrainer():
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
        self.weight_decay = cfg.gnn.train.weight_decay
        self.save_logits = cfg.gnn.train.save_logits

        self.n_heads = 3
        self.input_drop = 0.25
        self.attn_drop = 0.0
        self.edge_drop = 0.3
        self.no_attn_dst = True
        self.use_norm = False
        self.group = 2
        self.input_norm = 'T'
        self.seed = cfg.seed

        # ! Load data
        dataset, num_classes = load_data(self.dataset_name, use_dgl=True,
                                         use_text=False, seed=self.seed)
        data = dataset[0]

        self.train_mask = dataset.train_mask
        self.val_mask = dataset.val_mask
        self.test_mask = dataset.test_mask

        # Check Multi-label
        self.multi_label = (data.ndata['label'].dim() > 1 and data.ndata['label'].dtype == torch.float32)

        self.y = data.ndata['label'].to(self.device)
        if not self.multi_label:
            self.y = self.y.squeeze()

        self.num_nodes = data.num_nodes()
        self.num_classes = num_classes

        # ! Init gnn feature
        topk = 3 if self.dataset_name == 'pubmed' else 5
        if self.feature_type == 'ogb':
            print("Loading OGB features...")
            features = data.ndata['feat']
        elif self.feature_type == 'P':
            print("Loading top-k prediction features ...")
            features = load_gpt_preds(self.dataset_name, topk)
        # else:
        #     # Generalized loading for NS, E, R, KEA-I, TN, M, NM, etc.
        #     print(f"Loading pretrained LM features ({self.feature_type}) ...")
        #     LM_emb_path = f"prt_lm/{self.dataset_name}/{self.lm_model_name}_{self.feature_type}-seed{self.seed}.emb"
        #     print(f"LM_emb_path: {LM_emb_path}")
        #     features = torch.from_numpy(np.array(
        #         np.memmap(LM_emb_path, mode='r',
        #                   dtype=np.float16,
        #                   shape=(self.num_nodes, 768)))
        #     ).to(torch.float32)
        else:
            print(f"Loading pretrained LM features ({self.feature_type}) ...")
            LM_emb_path = f"prt_lm/{self.dataset_name}/{self.lm_model_name}_{self.feature_type}-seed{self.seed}.emb"
            print(f"LM_emb_path: {LM_emb_path}")

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
        if self.gnn_model_name == "RevGAT":
            self.model = RevGAT(in_feats=self.hidden_dim * topk if use_pred else self.features.shape[1],
                                n_classes=self.num_classes,
                                n_hidden=self.hidden_dim,
                                n_layers=self.num_layers,
                                n_heads=self.n_heads,
                                activation=torch.nn.Mish(),
                                dropout=self.dropout,
                                input_drop=self.input_drop,
                                attn_drop=self.attn_drop,
                                edge_drop=self.edge_drop,
                                use_attn_dst=not self.no_attn_dst,
                                use_symmetric_norm=self.use_norm,
                                group=self.group,
                                input_norm=self.input_norm == 'T',
                                use_pred=use_pred
                                ).to(self.device)
        else:
            exit(f'Error: model {self.gnn_model_name} not supported')

        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)

        print(f"\nNumber of parameters: {trainable_params}")
        self.ckpt = f"output/{self.dataset_name}/{self.gnn_model_name}.pt"
        self.stopper = EarlyStopping(
            patience=cfg.gnn.train.early_stop, path=self.ckpt) if cfg.gnn.train.early_stop > 0 else None

        if self.multi_label:
            self.loss_func = torch.nn.BCEWithLogitsLoss(reduction='mean')
        else:
            self.loss_func = torch.nn.CrossEntropyLoss(reduction='mean')

        from core.GNNs.gnn_utils import Evaluator
        self._evaluator = Evaluator(name=self.dataset_name)
        if self.multi_label:
            self.evaluator = lambda pred, labels: self._evaluator.eval(
                {"y_pred": pred, "y_true": labels})["micro_f1"]
        else:
            self.evaluator = lambda pred, labels: self._evaluator.eval(
                {"y_pred": pred.argmax(dim=-1, keepdim=True),
                 "y_true": labels.view(-1, 1)})["acc"]

    def _forward(self, data, feats):
        logits = self.model(data, feats)  # small-graph
        return logits

    def _train(self):
        # ! Shared
        self.model.train()
        self.optimizer.zero_grad()
        # ! Specific
        logits = self._forward(self.data, self.features)
        loss = self.loss_func(
            logits[self.train_mask], self.y[self.train_mask])
        train_acc = self.evaluator(
            logits[self.train_mask], self.y[self.train_mask])
        loss.backward()
        self.optimizer.step()

        return loss.item(), train_acc

    @torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        logits = self._forward(self.data, self.features)
        if self.multi_label:
            val_acc = self._evaluator.eval({"y_pred": logits[self.val_mask], "y_true": self.y[self.val_mask]})
            test_acc = self._evaluator.eval({"y_pred": logits[self.test_mask], "y_true": self.y[self.test_mask]})
        else:
            val_acc = self.evaluator(
                logits[self.val_mask], self.y[self.val_mask])
            test_acc = self.evaluator(
                logits[self.test_mask], self.y[self.test_mask])
        return val_acc, test_acc, logits

    @time_logger
    def train(self):
        # ! Training
        for epoch in range(self.epochs):
            if epoch <= 50 and self.gnn_model_name == 'RevGAT':
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr * epoch / 50
            t0, es_str = time(), ''
            loss, train_acc = self._train()
            val_acc, test_acc, _ = self._evaluate()
            val_score = val_acc['micro_f1'] if self.multi_label else val_acc
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