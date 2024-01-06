import torch
import torch.nn as nn

import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network
from torch_geometric.graphgym.models import GNNPreMP
from torch_geometric.graphgym.models.layer import LayerConfig

from ..utils import init_weights
from ..network.gnn import GNNLayer
from ..layer.dropout import Dropout


class MaskedFeaEncoder(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        self.mask_rate = cfg.graphmae.mask_rate
        self.replace_rate = cfg.graphmae.replace_rate

    def forward(self, batch):
        device = batch.x.device
        num_nodes = batch.num_nodes

        # random masking
        num_mask_nodes = int(self.mask_rate * num_nodes)
        batch.mask_nodes = torch.randperm(num_nodes,
                                          device=device)[:num_mask_nodes]

        if self.replace_rate > 0:
            num_noise_nodes = int(self.replace_rate * num_mask_nodes)
            noise_indices, token_indices = torch.randperm(
                num_mask_nodes, device=device).tensor_split(
                    (num_noise_nodes, ))
            token_nodes = batch.mask_nodes[token_indices]
            noise_nodes = batch.mask_nodes[noise_indices]
            noise_to_be_chosen = torch.randperm(
                num_nodes, device=device)[:num_noise_nodes]

            batch.x[noise_nodes] = batch.x[noise_to_be_chosen]
        else:
            token_nodes = batch.mask_nodes

        batch.x[token_nodes] = self.enc_mask_token

        return batch


@register_network('graphmae')
class GraphMAE(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        GNNStage = register.stage_dict[cfg.gnn.stage_type]
        Linear = register.layer_dict['linear']

        dim_cur = dim_in
        encoder_mods = []
        if cfg.gnn.layers_pre_mp > 0:
            encoder_mods.append(
                GNNPreMP(dim_cur, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp))
            dim_cur = cfg.gnn.dim_inner
        if cfg.gnn.layers_mp > 0:
            encoder_mods.append(
                GNNStage(dim_in=dim_cur,
                         dim_out=cfg.gnn.dim_inner,
                         num_layers=cfg.gnn.layers_mp))
            dim_cur = cfg.gnn.dim_inner
            if cfg.gnn.layer_type == 'custom_gatconv':
                dim_cur *= cfg.gnn.att_heads

        self.masked_enc = MaskedFeaEncoder(dim_in)
        self.encoder = nn.Sequential(*encoder_mods)
        self.encoder_to_decoder = Linear(
            LayerConfig(dim_in=dim_cur, dim_out=dim_out, has_bias=False))
        self.decoder = nn.Sequential(
            Dropout(cfg.gnn.in_drop),
            GNNLayer(dim_out,
                     dim_in,
                     has_act=False,
                     name=cfg.gae.decoder.layer_type,
                     heads=1,
                     att_drop=cfg.gnn.att_drop))

        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)

    def forward(self, batch):
        """"""
        # masked encode
        batch = self.masked_enc(batch)
        batch = self.encoder(batch)

        # remask
        batch = self.encoder_to_decoder(batch)
        if cfg.gae.decoder.layer_type != 'mlp':
            batch.x[batch.mask_nodes] = 0

        batch = self.decoder(batch)

        return batch.x

    def embed(self, batch):
        return self.encoder(batch).x
