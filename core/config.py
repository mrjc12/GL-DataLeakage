# core/config.py
import os
import argparse
from yacs.config import CfgNode as CN


def set_cfg(cfg):
    # ------------------------------------------------------------------------ #
    # Basic options
    # ------------------------------------------------------------------------ #
    cfg.dataset = 'cora'
    cfg.device = 0
    cfg.seed = None
    cfg.runs = 4

    # ------------------------------------------------------------------------ #
    # Data options 
    # ------------------------------------------------------------------------ #
    cfg.data = CN()
    cfg.data.root = 'processed_data'
    cfg.data.split = 'random'
    cfg.data.format = 'sbert'

    cfg.gnn = CN()
    cfg.lm = CN()

    # ------------------------------------------------------------------------ #
    # GNN Model options
    # ------------------------------------------------------------------------ #
    cfg.gnn.model = CN()
    cfg.gnn.model.name = 'GCN'
    cfg.gnn.model.num_layers = 3
    cfg.gnn.model.hidden_dim = 128

    # ------------------------------------------------------------------------ #
    # GNN Training options
    # ------------------------------------------------------------------------ #
    cfg.gnn.train = CN()
    cfg.gnn.train.weight_decay = 0.0
    cfg.gnn.train.epochs = 200
    cfg.gnn.train.feature_type = 'TA_P_E'
    cfg.gnn.train.early_stop = 50
    cfg.gnn.train.lr = 0.01
    cfg.gnn.train.wd = 0.0
    cfg.gnn.train.dropout = 0.0
    cfg.gnn.train.save_logits = True

    # ------------------------------------------------------------------------ #
    # LM Model options
    # ------------------------------------------------------------------------ #
    cfg.lm.model = CN()
    cfg.lm.model.name = 'microsoft/deberta-base'
    cfg.lm.model.feat_shrink = ""

    # ------------------------------------------------------------------------ #
    # LM Training options
    # ------------------------------------------------------------------------ #
    cfg.lm.train = CN()
    cfg.lm.train.batch_size = 9
    cfg.lm.train.grad_acc_steps = 1
    cfg.lm.train.lr = 2e-5
    cfg.lm.train.epochs = 4
    cfg.lm.train.warmup_epochs = 0.6
    cfg.lm.train.eval_patience = 50000
    cfg.lm.train.weight_decay = 0.0
    cfg.lm.train.dropout = 0.3
    cfg.lm.train.att_dropout = 0.1
    cfg.lm.train.cla_dropout = 0.4

    cfg.lm.train.text_type = 'TA'

    return cfg


def update_cfg(cfg, args_str=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="",
                        metavar="FILE", help="Path to config file")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    if isinstance(args_str, str):
        args = parser.parse_args(args_str.split())
    else:
        args = parser.parse_args()

    cfg = cfg.clone()

    if os.path.isfile(args.config):
        cfg.merge_from_file(args.config)

    cfg.merge_from_list(args.opts)

    return cfg


"""
    Global variable
"""
cfg = set_cfg(CN())
