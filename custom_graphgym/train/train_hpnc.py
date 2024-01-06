import time
from typing import Dict

from sklearn.cluster import KMeans
import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.model_builder import GraphGymModule
from torch_geometric.graphgym.register import loss_dict, register_train, metric_dict

from ..utils import calc_dim_out, load_centers


class HPNCModule(GraphGymModule):
    def __init__(self, dim_in, dim_out, cfg):
        super().__init__(dim_in, dim_out, cfg)
        self.model.set_centers(load_centers(cfg.share.dim_out, calc_dim_out()))

    def _shared_step(self, batch) -> Dict:
        x_rec, x_orig, pred_score, embed = self(batch)

        losses = dict.fromkeys(['fea_rec', 'edge_rec', 'clu', 'bal'],
                               torch.zeros(1, device=self.device))

        if cfg.hpnc.fea_rec_coeff:
            losses['fea_rec'], _ = loss_dict[cfg.hpnc.fea_rec_loss](
                x_rec[batch.mask_nodes], x_orig[batch.mask_nodes])
            losses['fea_rec'] *= cfg.hpnc.fea_rec_coeff

        if cfg.hpnc.edge_rec_coeff:
            losses['edge_rec'] = cfg.hpnc.edge_rec_coeff * loss_dict[
                cfg.hpnc.edge_rec_loss](embed, batch.edge_index)

        if cfg.hpnc.clu_coeff:
            losses['clu'] = cfg.hpnc.clu_coeff * loss_dict[cfg.hpnc.clu_loss](
                pred_score)

        if cfg.hpnc.bal_coeff:
            losses['bal'] = cfg.hpnc.bal_coeff * loss_dict[cfg.hpnc.bal_loss](
                pred_score)

        loss = sum(losses.values())

        custom_metrics = {}
        if cfg.hpnc.perform_kmeans:
            embed_pred = KMeans(cfg.share.dim_out, n_init='auto').fit_predict(
                                    embed.detach().cpu())
            embed_pred = torch.from_numpy(embed_pred)
            y_cpu = batch.y.cpu()
            for metric in ('accuracy', 'nmi', 'ari'):
                custom_metrics[f'km_{metric}'] = round(
                    metric_dict[metric]([embed_pred], [y_cpu], None),
                    cfg.round)
            # same prediction proportion of kmeans and hpnc
            custom_metrics['same_pred'] = round(
                metric_dict['accuracy']([embed_pred],
                                        [pred_score.argmax(dim=1).cpu()],
                                        None), cfg.round)

        step_end_time = time.time()
        return dict(loss=loss,
                    true=batch.y,
                    pred_score=pred_score,
                    step_end_time=step_end_time,
                    custom_metrics=custom_metrics)

    def training_step(self, batch, *args, **kwargs):
        return self._shared_step(batch)

    def validation_step(self, batch, *args, **kwargs):
        return self._shared_step(batch)

    def test_step(self, batch, *args, **kwargs):
        return self._shared_step(batch)


@register_train('train_hpnc')
def create_model(to_device=True, dim_in=None, dim_out=None) -> GraphGymModule:
    r"""Create model for graph machine learning.

    Args:
        to_device (bool, optional): Whether to transfer the model to the
            specified device. (default: :obj:`True`)
        dim_in (int, optional): Input dimension to the model
        dim_out (int, optional): Output dimension to the model
    """
    dim_in = cfg.share.dim_in if dim_in is None else dim_in
    dim_out = cfg.share.dim_out if dim_out is None else dim_out
    # binary classification, output dim = 1
    if 'classification' in cfg.dataset.task_type and dim_out == 2:
        dim_out = 1

    model = HPNCModule(dim_in, dim_out, cfg)
    return model
