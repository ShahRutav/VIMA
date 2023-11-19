import torch.nn as nn
import torch.nn.functional as F


class OneHotEmbedding(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, x):
        return F.one_hot(x, num_classes=self.n_classes)

    @property
    def output_dim(self):
        return self.n_classes
