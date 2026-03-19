import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn, einsum
from einops.layers.torch import Rearrange
import math
import numpy as np

class Channel2D3(nn.Module):
    def __init__(self):
        super().__init__()
        self.triangles = [
            (1, 4, -1, -1),
            (1, -1, -1, -1),
            (1, 2, 5, -1),
            (2, -1, -1, -1),
            (2, 3, 6, -1),
            (3, -1, -1, -1),
            (3, 7, -1, -1),
            (4, -1, -1, -1),
            (1, 4, 5, 8),
            (5, -1, -1, -1),
            (2, 5, 6, 9),
            (6, -1, -1, -1),
            (3, 6, 7, 10),
            (7, -1, -1, -1),# 三角形1 (倒三角)
            (4, 8, 11, -1),
            (8, -1, -1, -1),
            (5, 8, 9, 12),
            (9, -1, -1, -1),
            (6, 9, 10, 13),
            (10, -1, -1, -1),
            (7, 10, 14, -1),  # 三角形1 (倒三角)
            (11, -1, -1, -1),
            (8, 11, 12, 15),
            (12, -1, -1, -1),
            (9, 12, 13, 16),
            (13, -1, -1, -1),
            (10, 13, 14, 17),
            (14, -1, -1, -1),# 三角形1 (倒三角)
            (11, 15, 18, -1),
            (15, -1, -1, -1),
            (12, 15, 16, 19),
            (16, -1, -1, -1),
            (13, 16, 17, 20),
            (17, -1, -1, -1),
            (14, 17, 21, -1),# 三角形1 (倒三角)
            (18, -1, -1, -1),
            (15, 18, 19, 22),
            (19, -1, -1, -1),
            (16, 19, 20, 23),
            (20, -1, -1, -1),
            (17, 20, 21, 24),
            (21, -1, -1, -1),# 三角形1 (倒三角)
            (18, 22, -1, -1),
            (22, -1, -1, -1),
            (19, 22, 23, -1),
            (23, -1, -1, -1),
            (20, 23, 24, -1),
            (24, -1, -1, -1),
            (21, 24, -1, -1),
        ]

    def forward(self, x, b, l):  # x[128, 2, 160]
        B = torch.rand([b, 2, 7, 7, l])
        for i in range(1, 7 + 1):
            for j in range(1, 7 + 1):
                n = (i - 1) * 7 + j
                v = [0, 0, 0, 0]
                v[0], v[1], v[2], v[3] = self.triangles[n - 1]
                B[:, :, i - 1, j - 1, :]=0
                d=0
                for k in range(0, 3 + 1):
                    if(v[k]>0):
                        B[:, :, i - 1, j - 1, :]=B[:, :, i - 1, j - 1, :]+x[:, :, v[k] - 1, :]
                        d=d+1
                B[:, :, i - 1, j - 1, :] = B[:, :, i - 1, j - 1, :]/d
        return B
class Channel2D2(nn.Module):
    def __init__(self):
        super().__init__()
        self.triangles = [
            (1, 4, 5, -1),
            (1, 2, 5, -1),
            (2, 5, 6, -1),
            (2, 3, 6, -1),
            (3, 6, 7, -1),# 三角形1 (倒三角)
            (4, 5, 8, -1),
            (5, 8, 9, -1),
            (5, 6, 9, -1),
            (6, 9, 10, -1),
            (6, 7, 10, -1),# 三角形1 (倒三角)
            (8, 11, 12, -1),
            (8, 9, 12, -1),
            (9, 12, 13, -1),
            (9, 10, 13, -1),
            (10, 13, 14, -1),# 三角形1 (倒三角)
            (11, 12, 15, -1),
            (12, 15, 16, -1),
            (12, 13, 16, -1),
            (13, 16, 17, -1),
            (13, 14, 17, -1),# 三角形1 (倒三角)
            (15, 18, 19, -1),
            (15, 16, 19, -1),
            (16, 19, 20, -1),
            (16, 17, 20, -1),
            (17, 20, 21, -1),# 三角形1 (倒三角)
            (18, 19, 22, -1),
            (19, 22, 23, -1),
            (19, 20, 23, -1),
            (20, 23, 24, -1),
            (20, 21, 24, -1),# 三角形1 (倒三角)
        ]

    def forward(self, x, b, l):  # x[128, 2, 160]
        B = torch.rand([b, 2, 6, 5, l])
        for i in range(1, 6 + 1):
            for j in range(1, 5 + 1):
                n = (i - 1) * 5 + j
                v = [0, 0, 0, 0]
                v[0], v[1], v[2], v[3] = self.triangles[n - 1]
                B[:, :, i - 1, j - 1, :]=0
                d=0
                for k in range(0, 3 + 1):
                    if(v[k]>0):
                        B[:, :, i - 1, j - 1, :]=B[:, :, i - 1, j - 1, :]+x[:, :, v[k] - 1, :]
                        d=d+1
                B[:, :, i - 1, j - 1, :] = B[:, :, i - 1, j - 1, :]/d
        return B

class fNIRS_3D(nn.Module):  # n_class=3, sampling_point=sampling_points, dim=128, depth=6, heads=8, mlp_dim=64

    def __init__(self, n_class, dim, depth, heads, mlp_dim, channel, point, dim_head=40, emb_dropout=0.):#dim_head=40
        super().__init__()  # [128, 2, 258, 20]
        num_channels = 100
        dropout = 0

        self.chang_shape = nn.Sequential(  # [128, 2, 258, 20]
            Rearrange('b c h w  -> b c w h')  # [128, 2, 20, 258]
        )

        self.to_channel_embedding = nn.Sequential(  # [128, 2, 258, 20]
            nn.Conv3d(in_channels=2, out_channels=8, kernel_size=(6, 5, int(point ** 0.5)), stride=(6, 5, int(point ** 0.5))),  # [128, 2, 6, 2, 258]
            Rearrange('b c h w e -> b (w h) (c e)'),  # [128, 8, 1, 1, 20]->[128, 8, 20]
            nn.LayerNorm(8 * int(point ** 0.5)),
            nn.Linear(8 * int(point ** 0.5), n_class) # [128, 1, 160]
            )

        self.to_latent = nn.Identity()

        self.Channel2D2 = Channel2D2()

    def forward(self, img, mask=None):

        b, n, l, _ = img.squeeze(1).shape
        x2 = self.Channel2D2(self.chang_shape(img.squeeze(1)),
                             b, l)  # [128, 2, 258, 20]->[128, 2,20, 258]->[128, 2, 3, 2, 258]

        x2 = self.to_channel_embedding(x2)  # [128, 1, 160]
        k = torch.sigmoid(x2)
        x2 = x2 + k * x2

        x2 = self.to_latent(x2.squeeze(1))  # [128, 128]

        x4 = x2 #j * self.a

        return x4  # [128, 3]