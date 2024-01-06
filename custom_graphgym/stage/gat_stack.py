import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_stage
from ..network.gnn import GNNLayer
from ..layer.dropout import Dropout


@register_stage('gat_stack')
class GATStackStage(nn.Module):
    """
    Simple Stage that stack GNN layers

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        num_layers (int): Number of GNN layers
    """
    def __init__(self, dim_in, dim_out, num_layers):
        super().__init__()
        self.num_layers = num_layers
        d_in = dim_in
        for i in range(num_layers):
            if cfg.gnn.in_drop > 0:
                self.add_module('in_drop{}'.format(i),
                                Dropout(cfg.gnn.in_drop))
            if cfg.gnn.layer_type == 'custom_gatconv':
                layer = GNNLayer(d_in,
                                 dim_out,
                                 heads=cfg.gnn.att_heads,
                                 att_drop=cfg.gnn.att_drop)
                d_in = dim_out * cfg.gnn.att_heads
            else:
                layer = GNNLayer(d_in, dim_out)
                d_in = dim_out
            self.add_module('layer{}'.format(i), layer)

    def forward(self, batch):
        """"""
        for i, layer in enumerate(self.children()):
            batch = layer(batch)
        if cfg.gnn.l2norm:
            batch.x = F.normalize(batch.x, p=2, dim=-1)
        return batch
