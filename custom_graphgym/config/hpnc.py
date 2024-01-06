from yacs.config import CfgNode as CN

from torch_geometric.graphgym.register import register_config


@register_config('hpnc')
def set_cfg_hpnc(cfg):
    cfg.train.mode = 'standard'

    cfg.hpnc = CN()

    cfg.hpnc.backbone = 'graphmae'
    cfg.hpnc.rot_centers = True
    cfg.hpnc.perform_kmeans = False

    cfg.hpnc.clu_loss = 'dec'
    cfg.hpnc.clu_coeff = 1.0

    cfg.hpnc.fea_rec_loss = 'sce'
    cfg.hpnc.fea_rec_coeff = 1.0

    cfg.hpnc.bal_loss = 'bal_squ'
    cfg.hpnc.bal_coeff = 1.0

    cfg.hpnc.edge_rec_loss = 'edge_bce'
    cfg.hpnc.edge_rec_coeff = 1.0
