#!/usr/bin/env python
import pathlib
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

import click


class Prototype(torch.nn.Module):
    def __init__(self, nums, dim):
        super().__init__()
        self.param = torch.nn.Parameter(torch.empty((nums, dim)))
        torch.nn.init.xavier_uniform_(self.param.data)

    def forward(self):
        return F.normalize(self.param, dim=1)


def prototype_loss(prototypes):
    # Dot product of normalized prototypes is cosine similarity.
    product = torch.matmul(prototypes, prototypes.t()) + 1
    # Remove diagnonal from loss.
    product -= 2. * torch.diag(torch.diag(product))
    # Minimize maximum cosine similarity.
    loss = product.max(dim=1)[0]
    return loss.mean(), product.max()


def train(model, optim, scheduler, epoches):
    tr = trange(epoches)
    for _ in tr:
        loss, sep = prototype_loss(model())
        tr.set_postfix({
            'loss': loss.item(),
            'sep': sep.item(),
            'lr': scheduler.get_last_lr()[0]
        })

        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()


def seed_everything(seed: int):
    r"""Sets the seed for generating random numbers in PyTorch, numpy and
    Python.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@click.command()
@click.option('-n', '--num-class', type=int, required=True)
@click.option('-d', '--dim', type=int, required=True)
@click.option('--lr', type=float, default=1e-3, show_default=True)
@click.option('-m', '--momentum', type=float, default=0.9, show_default=True)
@click.option('-e', '--epochs', type=int, default=100000, show_default=True)
@click.option('-s', '--save-dir', default='prototypes', show_default=True)
@click.option('--gpu', type=int, default=None)
@click.option('--seed', type=int, default=None)
def main(num_class, dim, lr, momentum, gpu, epochs, save_dir, seed):
    if seed is not None:
        seed_everything(seed)

    device = torch.device(f'cuda:{gpu}' if gpu else 'cpu')
    model = Prototype(num_class, dim).to(device)

    optim = torch.optim.SGD(model.parameters(), lr, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs)
    train(model, optim, scheduler, epochs)

    model_file = pathlib.Path(
        save_dir,
        f'prototype_{num_class}c_{dim}d_{epochs}e_{int(time.time())}.pt')
    model_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model().detach().cpu(), model_file)
    print(f'Prototype saved to {model_file}')


if __name__ == "__main__":
    main()
