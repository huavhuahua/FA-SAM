# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from typing import Type


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        y = self.weight[:, None, None] * x
        # y = torch.mul(self.weight[:, None, None], x)
        x = y + self.bias[:, None, None]
        return x


class Adapter(nn.Module):
    def __init__(self, dim_features, mlp_ratio=0.25, parallel = True) -> None:
        super().__init__()
        self.parallel = parallel
        self.act = nn.GELU()
        dim_hidden_features = int(dim_features * mlp_ratio)
        self.D_fc1 = nn.Linear(dim_features, dim_hidden_features)
        self.D_fc2 = nn.Linear(dim_hidden_features, dim_features)

    def forward(self, x):
        # print('ADAPTER INPUT:', x.max(), x.min())
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)

        if self.parallel == True:
            x = x + xs
        else:
            x = xs

        # print('ADAPTER OUTPUT:', x.max(), x.min())
        return x