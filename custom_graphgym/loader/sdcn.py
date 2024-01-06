import os.path as osp
from typing import Callable, List, Optional

import torch
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loader
from torch_geometric.io import read_txt_array

from ..utils import build_transform


class SDCNDataset(InMemoryDataset):
    """
    https://arxiv.org/abs/2002.01633
    """

    url = 'https://github.com/bdy9527/SDCN/raw/master'

    def __init__(self,
                 root: str,
                 name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name.upper()

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'SDCN', self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'SDCN', self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        suffices = ['', '_label', '_graph']
        return [f'{self.name.lower()}{suffix}.txt' for suffix in suffices]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        dirs = {'data': ['', '_label'], 'graph': ['_graph']}
        for d, suffices in dirs.items():
            for suffix in suffices:
                download_url(f'{self.url}/{d}/{self.name.lower()}{suffix}.txt',
                             self.raw_dir)

    def process(self):
        x = read_txt_array(self.raw_paths[0])
        y = read_txt_array(self.raw_paths[1], dtype=torch.long)
        edge_index = read_txt_array(self.raw_paths[2],
                                    dtype=torch.long).t().contiguous()

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}()'


@register_loader('sdcn')
def load_sdcn(format, name, dataset_dir):
    if format != 'SDCN':
        return

    pre_transform = build_transform(cfg.dataset.pre_transform)
    transform = build_transform(cfg.dataset.transform)
    kwargs = {'transform': transform, 'pre_transform': pre_transform}

    if name in ['ACM', 'DBLP']:
        dataset = SDCNDataset(dataset_dir, name, **kwargs)
    else:
        raise ValueError('{} not support'.format(name))

    return dataset
