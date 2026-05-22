import torch
import numpy as np

from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, IntervalStrategy
from core.LMs.model import BertClassifier, BertClaInfModel
from core.data_utils.dataset import Dataset
from core.data_utils.load import load_data
from core.utils import init_path, time_logger


def compute_metrics(p):
    from sklearn.metrics import accuracy_score, f1_score
    pred, labels = p
    if len(labels.shape) > 1 and labels.shape[1] > 1:
        pred_bin = (pred > 0).astype(int)
        return {
            "accuracy": f1_score(y_true=labels, y_pred=pred_bin, average='micro', zero_division=0),
            "macro_f1": f1_score(y_true=labels, y_pred=pred_bin, average='macro', zero_division=0)
        }
    else:
        pred = np.argmax(pred, axis=1)
        accuracy = accuracy_score(y_true=labels, y_pred=pred)
        return {"accuracy": accuracy}


class LMTrainer():
    def __init__(self, cfg):
        self.dataset_name = cfg.dataset
        self.seed = cfg.seed

        self.model_name = cfg.lm.model.name
        self.feat_shrink = cfg.lm.model.feat_shrink

        self.weight_decay = cfg.lm.train.weight_decay
        self.dropout = cfg.lm.train.dropout
        self.att_dropout = cfg.lm.train.att_dropout
        self.cla_dropout = cfg.lm.train.cla_dropout
        self.batch_size = cfg.lm.train.batch_size
        self.epochs = cfg.lm.train.epochs
        self.warmup_epochs = cfg.lm.train.warmup_epochs
        self.eval_patience = cfg.lm.train.eval_patience
        self.grad_acc_steps = cfg.lm.train.grad_acc_steps
        self.lr = cfg.lm.train.lr

        self.text_type = cfg.lm.train.text_type

        self.output_dir = f'output/{self.dataset_name}/{self.model_name}_{self.text_type}-seed{self.seed}'
        self.ckpt_dir = f'prt_lm/{self.dataset_name}/{self.model_name}_{self.text_type}-seed{self.seed}'

        # Preprocess data
        data, num_classes, text = load_data(
            dataset=self.dataset_name, use_text=True, text_type=self.text_type, seed=self.seed)
        self.data = data
        self.num_nodes = data.y.size(0)
        self.n_labels = num_classes

        # Check Multi-label
        self.multi_label = (data.y.dim() > 1 and data.y.dtype == torch.float32)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        X = tokenizer(text, padding=True, truncation=True, max_length=512)

        dataset = Dataset(X, data.y.tolist())
        self.inf_dataset = dataset

        self.train_dataset = torch.utils.data.Subset(
            dataset, self.data.train_mask.nonzero().squeeze().tolist())
        self.val_dataset = torch.utils.data.Subset(
            dataset, self.data.val_mask.nonzero().squeeze().tolist())
        self.test_dataset = torch.utils.data.Subset(
            dataset, self.data.test_mask.nonzero().squeeze().tolist())

        # Define pretrained tokenizer and model
        bert_model = AutoModel.from_pretrained(self.model_name)
        self.model = BertClassifier(bert_model,
                                    n_labels=self.n_labels,
                                    feat_shrink=self.feat_shrink)

        # prev_ckpt = f'prt_lm/{self.dataset_name}/{self.model_name}.ckpt'
        # if self.use_gpt_str and os.path.exists(prev_ckpt):
        #     print("Initialize using previous ckpt...")
        #     self.model.load_state_dict(torch.load(prev_ckpt))

        self.model.config.dropout = self.dropout
        self.model.config.attention_dropout = self.att_dropout

        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)
        print(f"\nNumber of parameters: {trainable_params}")

    @time_logger
    def train(self):
        # Define training parameters
        eq_batch_size = self.batch_size * 4
        train_steps = self.num_nodes // eq_batch_size + 1
        eval_steps = self.eval_patience // eq_batch_size
        warmup_steps = int(self.warmup_epochs * train_steps)

        # Define Trainer
        args = TrainingArguments(
            output_dir=self.output_dir,
            do_train=True,
            do_eval=True,
            eval_steps=eval_steps,
            evaluation_strategy=IntervalStrategy.STEPS,
            save_steps=eval_steps,
            learning_rate=self.lr,
            weight_decay=self.weight_decay,
            save_total_limit=1,
            load_best_model_at_end=True,
            gradient_accumulation_steps=self.grad_acc_steps,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size * 8,
            warmup_steps=warmup_steps,
            num_train_epochs=self.epochs,
            dataloader_num_workers=1,
            fp16=True,
            dataloader_drop_last=True,
        )
        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=compute_metrics,
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        # Train pre-trained model
        self.trainer.train()
        torch.save(self.model.state_dict(), init_path(f"{self.ckpt_dir}.ckpt"))
        print(f'LM saved to {self.ckpt_dir}.ckpt')

    @time_logger
    @torch.no_grad()
    def eval_and_save(self):
        emb = np.memmap(init_path(f"{self.ckpt_dir}.emb"),
                        dtype=np.float16,
                        mode='w+',
                        shape=(self.num_nodes, self.feat_shrink if self.feat_shrink else 768))
        pred = np.memmap(init_path(f"{self.ckpt_dir}.pred"),
                         dtype=np.float16,
                         mode='w+',
                         shape=(self.num_nodes, self.n_labels))

        inf_model = BertClaInfModel(
            self.model, emb, pred, feat_shrink=self.feat_shrink)
        inf_model.eval()
        inference_args = TrainingArguments(
            output_dir=self.output_dir,
            do_train=False,
            do_predict=True,
            per_device_eval_batch_size=self.batch_size * 8,
            dataloader_drop_last=False,
            dataloader_num_workers=1,
            fp16_full_eval=True,
        )

        trainer = Trainer(model=inf_model, args=inference_args)
        trainer.predict(self.inf_dataset)

        from core.GNNs.gnn_utils import Evaluator
        _evaluator = Evaluator(name=self.dataset_name)

        if self.multi_label:
            def eval_func(mask): return _evaluator.eval({
                "y_true": self.data.y[mask],
                "y_pred": torch.from_numpy(pred[mask]).float(),
            })
        else:
            def evaluator(preds, labels): return _evaluator.eval({
                "y_true": torch.tensor(labels).view(-1, 1),
                "y_pred": torch.tensor(preds).view(-1, 1),
            })["acc"]

            def eval_func(mask): return evaluator(
                np.argmax(pred[mask], -1),
                self.data.y[mask]
            )

        train_res = eval_func(self.data.train_mask)
        val_res = eval_func(self.data.val_mask)
        test_res = eval_func(self.data.test_mask)

        if self.multi_label:
            print(
                f'[LM] Train Micro: {train_res["micro_f1"]:.4f}, Val Micro: {val_res["micro_f1"]:.4f}, Test Micro: {test_res["micro_f1"]:.4f}')
            print(
                f'[LM] Train Macro: {train_res["macro_f1"]:.4f}, Val Macro: {val_res["macro_f1"]:.4f}, Test Macro: {test_res["macro_f1"]:.4f}\n')
            return {'TrainMicro': train_res["micro_f1"], 'ValMicro': val_res["micro_f1"], 'TestMicro': test_res["micro_f1"], 'TestMacro': test_res["macro_f1"]}
        else:
            print(
                f'[LM] Train: {train_res:.4f}, Val: {val_res:.4f}, Test: {test_res:.4f}\n')
            return {'TrainAcc': train_res, 'ValAcc': val_res, 'TestAcc': test_res}