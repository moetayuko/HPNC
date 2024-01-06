import logging
import os

import custom_graphgym  # noqa, register custom modules
import torch
from yacs.config import CfgNode as CN

from torch_geometric import seed_everything
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (
    cfg,
    dump_cfg,
    load_cfg,
    set_cfg,
    set_out_dir,
    set_run_dir,
)
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.register import train_dict
from custom_graphgym.train.train import GraphGymDataModule
from torch_geometric.graphgym.utils.agg_runs import agg_runs
from torch_geometric.graphgym.utils.comp_budget import params_count
from custom_graphgym.utils import calc_dim_out, auto_select_device
from custom_graphgym.train.train import train


if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    # Load config file
    set_cfg(cfg)
    load_cfg(cfg, args)
    set_out_dir(cfg.out_dir, args.cfg_file)
    # hack
    cfg_orig = CN()
    with open(args.cfg_file, "r") as f:
        cfg_orig = cfg_orig.load_cfg(f)
    cfg.model.loss_fun = cfg_orig.model.loss_fun
    del cfg_orig
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    dump_cfg(cfg)
    # Repeat for different random seeds
    for i in range(args.repeat):
        set_run_dir(cfg.out_dir)
        set_printing()
        # Set configurations for each run
        seed_everything(cfg.seed)
        auto_select_device()
        # Set machine learning pipeline
        datamodule = GraphGymDataModule()
        if cfg.train.mode == 'standard':
            model = create_model(dim_out=calc_dim_out())
        else:
            model = train_dict[cfg.train.mode](dim_out=calc_dim_out())
        # Print model info
        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info('Num parameters: %s', cfg.params)
        # Start training
        train(model, datamodule, logger=True)
        cfg.seed += 1

    # Aggregate results from different seeds
    if cfg.model.type == 'hpnc':
        agg_func = custom_graphgym.utils.agg_train_runs
    else:
        agg_func = agg_runs
    agg_func(cfg.out_dir, cfg.metric_best)
    # When being launched in batch mode, mark a yaml as done
    if args.mark_done:
        os.rename(args.cfg_file, f'{args.cfg_file}_done')
