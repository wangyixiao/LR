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

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):  # x[128, 2, 160]
        x1 = self.fn(x, **kwargs)[:, :1, :] + x[:, :1, :]
        x2 = torch.cat((x1, x[:, 1:2, :]), dim=1)
        return x2


class Residualy(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):  # x[128, 2, 160]
        x1 = self.fn(x, **kwargs)
        x1 = x1 + x
        return x1


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):  # dim=40
        super().__init__()
        inner_dim = dim_head * heads  # 512
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5  # 0.125  （x**y）表示 x 的 y 次幂

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):  # x[128, 2, 160]
        b, n, _, h = *x.shape, self.heads  # 180 2 4
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 对张量进行分块chunk to_qkv[128, 2, 160*3] ->qkv3个[128, 2, 160]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
                      qkv)  # qkv3个[128, 2, 160] -> qkv3个[128, 4, 2, 40]

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # einsum爱因斯坦求和约定

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):  # mask=None
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([  # nn.ModuleList:任意 nn.Module 的子类加到这个 list 里面
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residualy(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:  # x[128, 2, 160]
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class STCT_3D(nn.Module):  # n_class=3, sampling_point=sampling_points, dim=128, depth=6, heads=8, mlp_dim=64

    def __init__(self, n_class, dim, depth, heads, mlp_dim, channel, point, dim_head=40, emb_dropout=0.):#dim_head=40
        super().__init__()  # [128, 2, 258, 20]
        num_channels = 100
        dropout = 0

        self.chang_shape = nn.Sequential(  # [128, 2, 258, 20]
            Rearrange('b c h w  -> b c w h')  # [128, 2, 20, 258]
        )

        self.to_time_embedding = nn.Sequential(  # [128, 2, ]
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(point, 1), stride=(point, 1)),  # [128, 8, 1, 20]
            Rearrange('b c h w  -> b h (c w)'),  # [128, 1, 160]
            nn.Linear(channel*8, dim),  # [128, 1, 160]
            nn.LayerNorm(dim))

        self.to_channel_embedding = nn.Sequential(  # [128, 2, 258, 20]
            nn.Conv3d(in_channels=2, out_channels=8, kernel_size=(7, 7, int(point ** 0.5)), stride=(7, 7, int(point ** 0.5))),  # [128, 2, 6, 2, 258]
            Rearrange('b c h w e -> b (w h) (c e)'),  # [128, 8, 1, 1, 20]->[128, 8, 20]
            nn.LayerNorm(8 * int(point ** 0.5)),
            nn.Linear(8 * int(point ** 0.5), n_class) # [128, 1, 160]
            )

        self.pos_embedding_time = nn.Parameter(torch.randn(1, num_channels + 1, dim))
        self.cls_token_time = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer_time = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_latent = nn.Identity()  # identity表示该字段的值会自动更新，不需要我们维护

        self.mlp_head1 = nn.Sequential(
            nn.LayerNorm(dim*2),
            nn.Linear(dim*2, n_class))

        self.Channel2D3 = Channel2D3()

    def forward(self, img, mask=None):

        x1 = self.to_time_embedding(img.squeeze(1))  # [128, 1, 160]

        k = torch.sigmoid(x1)
        x1 = x1 + k * x1

        b, n, _ = x1.shape

        cls_tokens = repeat(self.cls_token_time, '() n d -> b n d', b=b)
        x4 = torch.cat((cls_tokens, x1), dim=1)

        b, n, _ = x4.shape

        x4 += self.pos_embedding_time[:, :n]  # [128, 2, 160]
        x4 = self.dropout(x4)
        x4 = self.transformer_time(x4, mask)  # [128, 2, 160]

        x4 = x4[:, 0]  # [128, 128]

        x4 = torch.cat((x4, x1.squeeze(1)), dim=1)
        x4 = self.to_latent(x4)  # [128, 128]
        x4 = self.mlp_head1(x4)

        b, n, l, _ = img.squeeze(1).shape
        x2 = self.Channel2D3(self.chang_shape(img.squeeze(1)),
                             b, l)  # [128, 2, 258, 20]->[128, 2,20, 258]->[128, 2, 3, 2, 258]

        x2 = self.to_channel_embedding(x2)  # [128, 1, 160]
        k = torch.sigmoid(x2)
        x2 = x2 + k * x2

        x2 = self.to_latent(x2.squeeze(1))  # [128, 128]

        x4 = x4+torch.sigmoid(x2) #j * self.a

        return x4  # [128, 3]