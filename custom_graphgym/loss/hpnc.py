import torch
import torch.nn.functional as F

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss
from torch_geometric.nn.models import GAE

EPS = 1e-15


def is_hpnc():
    TRAINERS = ['train_hpnc']
    return cfg.train.mode in TRAINERS


@torch.no_grad()
def auxiliary_distribution(q):
    p = q**2 / q.sum(dim=0, keepdim=True)
    p /= p.sum(dim=1, keepdim=True)
    return p


@register_loss('dec')
def loss_dec(q, *kwargs):
    if is_hpnc():
        p = auxiliary_distribution(q)
        loss = F.kl_div(q.log(), p, reduction='batchmean')  # KL(P || Q)
        return loss


@register_loss('edge_bce')
def loss_recon_edge_bce(z, edge_index):
    if is_hpnc():
        gae = GAE(torch.empty(0))
        loss = gae.recon_loss(z, edge_index)
        return loss


def entropy(p):
    p = p.clamp(EPS)
    ent = p * p.log()

    if ent.ndim == 1:
        return -ent.sum()
    elif ent.ndim == 2:
        return -ent.sum(dim=1).mean()
    else:
        raise ValueError(f'Probability is {ent.ndim}-d')


@register_loss('bal_ent')
def loss_balance_entropy(prob, *kwargs):
    if is_hpnc():
        # return negative entropy to maximize it
        return -entropy(prob)


@register_loss('rim')
def loss_rim(prob, *kwargs):
    if is_hpnc():
        return entropy(prob)


@register_loss('bal_rim')
def loss_balance_rim(prob, *kwargs):
    if is_hpnc():
        p_ave = prob.mean(dim=0)
        # return negative entropy to maximize it
        return -entropy(p_ave)
