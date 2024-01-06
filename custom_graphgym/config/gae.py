from yacs.config import CfgNode as CN

from torch_geometric.graphgym.register import register_config


@register_config('gae')
def set_cfg_gae(cfg):
    cfg.gae = CN()

    cfg.gae.decoder = CN()
    cfg.gae.decoder.layer_type = 'gat'
