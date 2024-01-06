from torch_geometric.graphgym.register import register_config


@register_config('custom_gnn')
def set_cfg_gnn(cfg):
    cfg.gnn.in_drop = 0.0
    cfg.gnn.att_drop = 0.0
