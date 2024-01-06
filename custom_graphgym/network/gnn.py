from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models import GeneralLayer
from torch_geometric.graphgym.models.layer import new_layer_config


def GNNLayer(dim_in, dim_out, has_act=True, name=None, **kwargs):
    """
    Wrapper for a GNN layer

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        has_act (bool): Whether has activation function after the layer

    """
    if name is None:
        name = cfg.gnn.layer_type
    layer_config = new_layer_config(dim_in,
                                    dim_out,
                                    1,
                                    has_act=has_act,
                                    has_bias=False,
                                    cfg=cfg)
    for key, value in kwargs.items():
        if hasattr(layer_config, key):
            setattr(layer_config, key, value)
    return GeneralLayer(name, layer_config=layer_config, **kwargs)
