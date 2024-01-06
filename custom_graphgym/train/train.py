from typing import Any, Dict, Optional
import warnings

import torch
from torch.utils.data import DataLoader
from torch_geometric.data.lightning.datamodule import LightningDataModule
from torch_geometric.graphgym import create_loader
from torch_geometric.graphgym.checkpoint import get_ckpt_dir
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.imports import pl
from torch_geometric.graphgym.logger import LoggerCallback
from torch_geometric.graphgym.model_builder import GraphGymModule


class GraphGymDataModule(LightningDataModule):
    def __init__(self):
        self.loaders = create_loader()
        super().__init__(has_val=False, has_test=False)

    def train_dataloader(self) -> DataLoader:
        return self.loaders[0]


class MyLoggerCallback(LoggerCallback):
    def _get_stats(
        self,
        epoch_start_time: int,
        outputs: Dict[str, Any],
        trainer: 'pl.Trainer',
    ) -> Dict:
        stats = super()._get_stats(epoch_start_time, outputs, trainer)
        if 'custom_metrics' in outputs:
            stats = stats | outputs['custom_metrics']
        return stats


def train(model: GraphGymModule, datamodule, logger: bool = True,
          trainer_config: Optional[dict] = None):
    warnings.filterwarnings('ignore', '.*use `CSVLogger` as the default.*')

    callbacks = []
    if logger:
        callbacks.append(MyLoggerCallback())
    if cfg.train.enable_ckpt:
        ckpt_cbk = pl.callbacks.ModelCheckpoint(
            dirpath=get_ckpt_dir(), every_n_epochs=cfg.train.ckpt_period, save_last=True)
        callbacks.append(ckpt_cbk)

    trainer_config = trainer_config or {}
    trainer = pl.Trainer(
        **trainer_config,
        enable_checkpointing=cfg.train.enable_ckpt,
        callbacks=callbacks,
        default_root_dir=cfg.out_dir,
        max_epochs=cfg.optim.max_epoch,
        accelerator='cuda',
        enable_progress_bar=False,
        check_val_every_n_epoch=cfg.train.eval_period,
        devices='auto' if not torch.cuda.is_available() else cfg.devices,
        strategy=pl.strategies.SingleDeviceStrategy(cfg.accelerator)
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path='last')
