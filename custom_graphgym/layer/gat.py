import torch.nn as nn

import torch_geometric as pyg
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.register import register_layer


@register_layer('custom_gatconv')
class GATConv(nn.Module):
    """
    Graph Attention Network (GAT) layer
    """
    def __init__(self,
                 layer_config: LayerConfig,
                 heads=None,
                 att_drop=0.0,
                 **kwargs):
        super().__init__()
        if heads is None:
            heads = cfg.gnn.att_heads
        self.model = pyg.nn.GATConv(layer_config.dim_in,
                                    layer_config.dim_out,
                                    bias=layer_config.has_bias,
                                    heads=heads,
                                    dropout=att_drop)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch
