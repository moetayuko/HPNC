from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loader
from torch_geometric.datasets import (PPI, Amazon, Coauthor, KarateClub,
                                      MNISTSuperpixels, Planetoid, QM7b,
                                      TUDataset, AttributedGraphDataset)
import torch_geometric.transforms as T
from ..utils import build_transform


@register_loader('pyg')
def load_pyg(format, name, dataset_dir):
    """
    Load PyG dataset objects. (More PyG datasets will be supported)

    Args:
        name (string): dataset name
        dataset_dir (string): data directory

    Returns: PyG dataset object

    """
    if format != 'PyG':
        return

    pre_transform = build_transform(cfg.dataset.pre_transform)
    transform = build_transform(cfg.dataset.transform)
    kwargs = {'transform': transform, 'pre_transform': pre_transform}

    dataset_dir = '{}/{}'.format(dataset_dir, name)
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(dataset_dir, name, **kwargs)
    elif name[:3] == 'TU_':
        # TU_IMDB doesn't have node features
        if name[3:] == 'IMDB':
            name = 'IMDB-MULTI'
            dataset = TUDataset(dataset_dir, name, transform=T.Constant())
        else:
            dataset = TUDataset(dataset_dir, name[3:], **kwargs)
    elif name == 'Karate':
        dataset = KarateClub(transform=transform)
    elif 'Coauthor' in name:
        if 'CS' in name:
            dataset = Coauthor(dataset_dir, name='CS', **kwargs)
        else:
            dataset = Coauthor(dataset_dir, name='Physics', **kwargs)
    elif 'Amazon' in name:
        if 'Computers' in name:
            dataset = Amazon(dataset_dir, name='Computers', **kwargs)
        else:
            dataset = Amazon(dataset_dir, name='Photo', **kwargs)
    elif name == 'MNIST':
        dataset = MNISTSuperpixels(dataset_dir, **kwargs)
    elif name == 'PPI':
        dataset = PPI(dataset_dir, **kwargs)
    elif name == 'QM7b':
        dataset = QM7b(dataset_dir, **kwargs)
    elif name == 'Wiki':
        dataset = AttributedGraphDataset(dataset_dir, name, **kwargs)
    else:
        raise ValueError('{} not support'.format(name))

    return dataset
