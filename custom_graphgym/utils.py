import logging
from numbers import Number
import os
import pathlib

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.utils.agg_runs import agg_dict_list, is_seed, join_list
from torch_geometric.graphgym.utils.io import (
    dict_list_to_json,
    dict_list_to_tb,
    dict_to_json,
    json_to_dict_list,
    makedirs_rm_exist,
)
import torch_geometric.nn as pyg_nn
import torch_geometric.transforms as T
from torch_geometric.graphgym.utils.device import get_gpu_memory_map

try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None


def init_weights(m):
    r"""
    Performs weight initialization

    Args:
        m (nn.Module): PyTorch module

    """
    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, pyg_nn.dense.Linear) or isinstance(m, nn.Linear):
        m.weight.data = nn.init.xavier_normal_(
            m.weight.data, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            m.bias.data.zero_()


def load_centers(n_clusters, in_dim, root='prototypes', device=None):
    root = pathlib.Path(root)
    pt_files = root.glob(f'prototype_{n_clusters}c_{in_dim}d_*.pt')
    pt_files = sorted(pt_files)
    if len(pt_files) == 0:
        raise ValueError(
            f"Can't find prototype of {n_clusters}c, {in_dim}d in {root}")

    pt_file = pt_files[-1]
    logging.info(f'Loading centers from {pt_file}')

    return torch.load(pt_file, map_location=device)


def calc_dim_out():
    dim_out = cfg.gnn.dim_inner
    if cfg.gnn.layer_type == 'custom_gatconv':
        dim_out *= cfg.gnn.att_heads
    return dim_out


def agg_train_runs(dir, metric_best='auto'):
    r'''
    Aggregate over different random seeds of a single experiment

    Args:
        dir (str): Directory of the results, containing 1 experiment
        metric_best (str, optional): The metric for selecting the best
        validation performance. Options: auto, accuracy, auc.

    '''
    results = {'train': None}
    results_best = {'train': None}
    split = 'train'
    for seed in os.listdir(dir):
        if is_seed(seed):
            dir_seed = os.path.join(dir, seed)

            if split in os.listdir(dir_seed):
                dir_split = os.path.join(dir_seed, split)
                fname_stats = os.path.join(dir_split, 'stats.json')
                stats_list = json_to_dict_list(fname_stats)
                if metric_best == 'auto':
                    metric = 'auc' if 'auc' in stats_list[0] else 'accuracy'
                else:
                    metric = metric_best
                performance_np = np.array(  # noqa
                    [stats[metric] for stats in stats_list])
                best_epoch = \
                    stats_list[
                        eval("performance_np.{}()".format(cfg.metric_agg))][
                        'epoch']
                print(best_epoch)

                stats_best = [
                    stats for stats in stats_list
                    if stats['epoch'] == best_epoch
                ][0]
                print(stats_best)

                # drop non-numeric values
                for stats in stats_list:
                    for k, v in list(stats.items()):
                        if not isinstance(v, Number):
                            del stats[k]

                stats_list = [[stats] for stats in stats_list]
                if results[split] is None:
                    results[split] = stats_list
                else:
                    results[split] = join_list(results[split], stats_list)
                if results_best[split] is None:
                    results_best[split] = [stats_best]
                else:
                    results_best[split] += [stats_best]
    results = {k: v for k, v in results.items() if v is not None}  # rm None
    results_best = {k: v
                    for k, v in results_best.items()
                    if v is not None}  # rm None
    for i in range(len(results[split])):
        results[split][i] = agg_dict_list(results[split][i])
    results_best[split] = agg_dict_list(results_best[split])
    # save aggregated results
    for key, value in results.items():
        dir_out = os.path.join(dir, 'agg', key)
        makedirs_rm_exist(dir_out)
        fname = os.path.join(dir_out, 'stats.json')
        dict_list_to_json(value, fname)

        if cfg.tensorboard_agg:
            if SummaryWriter is None:
                raise ImportError(
                    'Tensorboard support requires `tensorboardX`.')
            writer = SummaryWriter(dir_out)
            dict_list_to_tb(value, writer)
            writer.close()
    for key, value in results_best.items():
        dir_out = os.path.join(dir, 'agg', key)
        fname = os.path.join(dir_out, 'best.json')
        dict_to_json(value, fname)
        print(value)
    logging.info('Results aggregated across runs saved in {}'.format(
        os.path.join(dir, 'agg')))


def build_transform(transform):
    transforms = []

    if transform == 'none':
        pass
    elif transform == 'norm_feat':
        transforms.append(T.NormalizeFeatures())
    else:
        raise ValueError(f'Unknown transform {transform}')

    return T.Compose(transforms)


def auto_select_device(memory_max=8000, memory_bias=200, strategy='random'):
    r'''
    Auto select device for the experiment. Useful when having multiple GPUs.
    Args:
        memory_max (int): Threshold of existing GPU memory usage. GPUs with
        memory usage beyond this threshold will be deprioritized.
        memory_bias (int): A bias GPU memory usage added to all the GPUs.
        Avoild dvided by zero error.
        strategy (str, optional): 'random' (random select GPU) or 'greedy'
        (greedily select GPU)
    '''
    if cfg.accelerator != 'cpu' and torch.cuda.is_available():
        if cfg.accelerator == 'auto':
            memory_raw = get_gpu_memory_map()
            if strategy == 'greedy' or np.all(memory_raw > memory_max):
                cuda = np.argmin(memory_raw)
                logging.info('GPU Mem: {}'.format(memory_raw))
                logging.info(
                    'Greedy select GPU, select GPU {} with mem: {}'.format(
                        cuda, memory_raw[cuda]))
            elif strategy == 'random':
                memory = 1 / (memory_raw + memory_bias)
                memory[memory_raw > memory_max] = 0
                gpu_prob = memory / memory.sum()
                cuda = np.random.choice(len(gpu_prob), p=gpu_prob)
                logging.info('GPU Mem: {}'.format(memory_raw))
                logging.info('GPU Prob: {}'.format(gpu_prob.round(2)))
                logging.info(
                    'Random select GPU, select GPU {} with mem: {}'.format(
                        cuda, memory_raw[cuda]))

            cfg.accelerator = 'cuda:{}'.format(cuda)
    else:
        cfg.accelerator = 'cpu'
