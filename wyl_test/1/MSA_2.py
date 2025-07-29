import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from torch import einsum

x = torch.randn(2, 3, 8, 8)  # 形状(batch_size, channels, H, W)
print("原始输入：", x)
print("-----")

class Attention(nn.Module):
    ''' dim和inner_dim，就是每个特征的维度
        heads就是头的个数，dim_head就是每个头的特征维度
        scale就是Q和K矩阵相乘后还要做的缩放值
        attend就是Q和K相乘并缩放后，还要再经过一道softmax层，以便后续与V相乘
        to_qkv相当于MSA_1.py中，三个线性层linear_q, linear_k, linear_v的合并写法
        to_out是对最后拼接的结果再做一次线性变换
    '''

    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        dim_head = dim // heads
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** (-0.5)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    ''' 输入x是二维特征图，形状(batch_size, channels, H, W) '''

    def forward(self, x):
        b, c, h, w, heads = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)

attn = Attention(dim=3)
print("加权信息：", attn(x))