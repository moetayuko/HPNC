import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    confusion_matrix,
    normalized_mutual_info_score,
)
from sklearn.utils.multiclass import unique_labels
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_metric


def compute_assignment(y_true, y_pred):
    labels = unique_labels(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_indices, col_indices = linear_sum_assignment(cm, maximize=True)
    return labels, cm, row_indices, col_indices


def get_pred(y_pred):
    if y_pred.ndim == 1:
        pass
    elif y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)
    else:
        raise ValueError('Wrong dimension')
    return y_pred


@register_metric('accuracy')
def clustering_accuracy(y_true, y_pred, task_type):
    y_true, y_pred = torch.cat(y_true), torch.cat(y_pred)
    y_pred = get_pred(y_pred)
    _, cm, row_indices, col_indices = compute_assignment(y_true, y_pred)
    acc = cm[row_indices, col_indices].sum().astype(float) / np.sum(cm)
    return round(acc, cfg.round)


@register_metric('nmi')
def nmi(y_true, y_pred, task_type):
    y_true, y_pred = torch.cat(y_true), torch.cat(y_pred)
    y_pred = get_pred(y_pred)
    nmi_score = normalized_mutual_info_score(y_true, y_pred)
    return round(nmi_score, cfg.round)


@register_metric('ari')
def ari(y_true, y_pred, task_type):
    y_true, y_pred = torch.cat(y_true), torch.cat(y_pred)
    y_pred = get_pred(y_pred)
    ari_score = adjusted_rand_score(y_true, y_pred)
    return round(ari_score, cfg.round)
