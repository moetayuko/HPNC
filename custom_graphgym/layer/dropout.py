import torch.nn as nn


class Dropout(nn.Dropout):
    def forward(self, batch):
        batch.x = super().forward(batch.x)
        return batch
