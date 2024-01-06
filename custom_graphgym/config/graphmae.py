from yacs.config import CfgNode as CN

from torch_geometric.graphgym.register import register_config


@register_config('graphmae')
def set_cfg_graphmae(cfg):
    cfg.graphmae = CN()

    cfg.graphmae.mask_rate = 0.5
    cfg.graphmae.replace_rate = 0.05
