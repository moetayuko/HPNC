import torch
import torch.nn.functional as F

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss
from .hpnc import is_hpnc


@register_loss('sce')
def loss_sce(pred, true):
    if cfg.model.loss_fun == 'sce' or is_hpnc():
        loss = (1 - F.cosine_similarity(pred, true)).pow_(cfg.sce.power)

        reduction = cfg.model.size_average
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        else:
            raise ValueError(f'Unknown reduction {reduction}')

        return loss, pred
