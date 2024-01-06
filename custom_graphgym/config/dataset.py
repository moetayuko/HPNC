from torch_geometric.graphgym.register import register_config


@register_config('custom_dataset')
def set_cfg_dataset(cfg):
    cfg.dataset.pre_transform = 'none'
