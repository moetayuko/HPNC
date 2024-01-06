import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.graphgym.models.head  # noqa, register module
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import network_dict, register_network


@register_network('hpnc')
class HPNC(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.ae = network_dict[cfg.hpnc.backbone](dim_in=dim_in,
                                                  dim_out=dim_out)
        self.register_buffer('_centers', torch.empty(cfg.share.dim_out, dim_out))

        if cfg.hpnc.rot_centers:
            rot = nn.Linear(dim_out, dim_out, bias=False)
            nn.init.eye_(rot.weight)
            self.rot = nn.utils.parametrizations.orthogonal(rot)
        else:
            self.rot = nn.Identity()

    def set_centers(self, centers):
        self._centers.copy_(centers)

    @property
    def centers(self):
        return self.rot(self._centers)

    def forward(self, batch):
        """"""
        x_orig = batch.x.clone()
        x_rec = self.ae(batch)

        batch.x = x_orig
        prob, embed = self.forward_prob(batch)

        return x_rec, x_orig, prob, embed

    def forward_prob(self, batch):
        embed = self.ae.embed(batch)
        embed = F.normalize(embed)

        prob = embed @ self.centers.T
        prob = F.softmax(prob, dim=1)

        return prob, embed
