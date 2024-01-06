from yacs.config import CfgNode as CN

from torch_geometric.graphgym.register import register_config


@register_config('sce')
def set_cfg_sce(cfg):
    cfg.sce = CN()

    cfg.sce.power = 3.0
