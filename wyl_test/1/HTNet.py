from functools import partial
import torch
from torch import nn, einsum

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce


def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else ((val,) * depth)


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mlp_mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mlp_mult, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        dim_head = dim // heads
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

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


def Aggregate(dim, dim_out):
    return nn.Sequential(
        nn.Conv2d(dim, dim_out, 3, padding=1),
        LayerNorm(dim_out),
        nn.MaxPool2d(3, stride=2, padding=1)
    )


class Transformer(nn.Module):
    def __init__(self, dim, seq_len, depth, heads, mlp_mult, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.pos_emb = nn.Parameter(torch.randn(seq_len))

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_mult, dropout=dropout))
            ]))

    def forward(self, x):
        *_, h, w = x.shape

        pos_emb = self.pos_emb[:(h * w)]
        pos_emb = rearrange(pos_emb, '(h w) -> () () h w', h=h, w=w)
        x = x + pos_emb

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class HTNet(nn.Module):
    def __init__(
            self,
            *,
            image_size,
            patch_size,
            num_classes,
            dim,
            heads,
            num_hierarchies,
            block_repeats,
            mlp_mult=4,
            channels=3,
            dim_head=64,
            dropout=0.
    ):
        super().__init__()
        assert (image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.')
        patch_dim = channels * patch_size ** 2
        fmap_size = image_size // patch_size
        blocks = 2 ** (num_hierarchies - 1)

        seq_len = (fmap_size // blocks) ** 2  # 每个block中包含几个patch，那么seq_len就是几，因为一个patch相当于一个词
        hierarchies = list(reversed(range(num_hierarchies)))  # [2, 1, 0]
        ''' 假设是3个层级，mults分别是1、2、4？感觉不太对啊，感觉应该是1块、4块、8块啊。 '''
        mults = [2 ** i for i in reversed(hierarchies)]  # [2^0, 2^1, 2^2] = [1, 2, 4]

        ''' layer_heads 是干啥的？
            假设heads=3，那么layer_heads = [3, 6, 12]，这是什么意思？第三层用12个头？ '''
        layer_heads = list(map(lambda t: t * heads, mults))
        ''' layer_dims 也不太懂 '''
        layer_dims = list(map(lambda t: t * dim, mults))
        last_dim = layer_dims[-1]
        ''' * 把layer_dims各元素解包，最后再拼上一个[-1] '''
        layer_dims = [*layer_dims, layer_dims[-1]]
        dim_pairs = zip(layer_dims[:-1], layer_dims[1:])

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w', p1=patch_size, p2=patch_size),
            nn.Conv2d(patch_dim, layer_dims[0], 1)
        )

        block_repeats = cast_tuple(block_repeats, num_hierarchies)
        self.layers = nn.ModuleList([])

        for level, heads, (dim_in, dim_out), block_repeat in zip(hierarchies, layer_heads, dim_pairs, block_repeats):
            is_last = level == 0
            depth = block_repeat
            self.layers.append(nn.ModuleList([
                Transformer(dim_in, seq_len, depth, heads, mlp_mult, dropout),
                Aggregate(dim_in, dim_out) if not is_last else nn.Identity()
            ]))

        self.mlp_head = nn.Sequential(
            LayerNorm(last_dim),
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(last_dim, num_classes)
        )

        def forward(self, img):
            x = self.to_patch_embedding
