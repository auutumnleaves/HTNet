from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
from vit_pytorch.cct import DropPath

"""
    Patch Embedding 对应代码实现
"""


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        '''
          img_size - 图像大小
          patch_size - 每个patch的大小
          in_chans - 输入通道数
          embed_dim - 词嵌入维度（每个token向量的长度）
          norm_layer - 可选LayerNorm
        '''
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)  # 把输入的大小化为二维元组
        self.img_size = img_size
        self.patch_size = patch_size  # 类的示例属性赋值
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # 网格大小，取[0]和[1]是因为size是元组
        self.num_patches = self.grid_size[0] * self.grid_size[1]  # 总共有多少块patch

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size)  # [3, 224, 224]  --> [768, 14, 14]
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()  # 从形参传入norm_layer，若不为空则使用之，若为空则保持不变

    def forward(self, x):
        B, C, H, W = x.shape  # batch, channel, height, weight
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"输入图像的大小{H}*{W}与模型期望大小{self.img_size[0]}*{self.img_size[1]}不匹配"
        ''' [B, 3, 224, 224]  --> [B, 768, 14, 14]
            flatten(2)相当于在第一个14处（即下标为2处）及以后作展平 --> [B, 768, 196]
            再做transpose(1, 2)维度的交换，把下标为1和2的维度互换 --> [B, 196, 768] '''
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)  # 若有则做，若无则不做
        return x


"""
    Attention 代码实现
"""


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, atte_drop_ratio=0., proj_drop_ratio=0.):
        '''
            dim - 输入的token的维度（注，不是序列长度），此处应为768
            num_heads - 多头注意力机制的头数，为8
            qkv_bias - qkv的偏置，默认False。具体来说，是使用线性层生成QKV矩阵时，是否添加偏置项，为False就是不添加
            qk_scale - qk的缩放因子，默认None。用于缩放QK的系数，如果为None则使用1/sqrt(head_dim) -- head_dim是总维度均分到每个单头上的维度
            atte_drop_ratio - 注意力分数的dropout的比例，防止过拟合
            proj_drop_ratio - 最终线性投影层的dropout比例
        '''
        super().__init__()
        self.num_heads = num_heads  # 注意力头数
        head_dim = dim // num_heads  # 每个头的维度
        self.scale = qk_scale or head_dim ** -0.5
        ''' 通过全连接层学习生成QKV
            有的代码会写三个单独的层分别生成Q,K,V，也行
            但是像这里一次生成的好处是能并行计算，且参数量会少一些 '''
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.att_drop = nn.Dropout(atte_drop_ratio)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        ''' 将每个单独head得到的输出进行concat拼接，然后通过线性变换映射一下 '''
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape  # batch, num_patch+1(+1是cls token), embed_dim
        ''' [B, N, 3 * C] --> [B, N, 3, num_heads, C // num_heads] --> [3, B, num_heads, N, C // num_heads]
            方便之后做运算 '''
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 用切片拿到Q,K,V，形状[B, num_heads, N, C // num_heads]
        ''' Q和K的转置作点积，并进行缩放，得到注意力分数
            q - [B, num_heads, N, C // num_heads]
            k的转置 - [B, num_heads, C // num_heads, N]
            点积的结果 - [B, num_heads, N, N] '''
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)  # 每行的数据作了softmax，和为1
        ''' 注意力权重对V进行加权求和
            attn - [B, num_heads, N, N]
            v - [B, num_heads, N, C // num_heads]
            点积的结果 - [B, num_heads, N, C // num_heads]
            transpose - [B, N, num_heads, C // num_heads]
            reshape - [B, N, C] （最后两个维度合并成了C，回到了总维度）'''
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)  # 再通过一个线性变换映射层
        x = self.proj_drop(x)  # 再过一个dropout层，防止过拟合

        return x


"""
   Transformer Encoder示意图中的MLP具体实现
"""


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_feature=None, out_features=None, act_layer=nn.GELU, drop=0.):
        '''
            in_features - 输入维度
            hidden_feature - 隐藏层维度，通常为输入维度的4倍
            out_features - 通常与输入维度相等
            act_layer - 激活函数
            drop - Dropout层丢弃率
        '''
        super().__init__()
        out_features = out_features or in_features  # 如果没有传输出维度，则默认等于输入维度
        hidden_feature = hidden_feature or in_features
        self.fc1 = nn.Linear(in_features, hidden_feature)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_feature, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)  # 第一个全连接层
        x = self.act(x)  # 激活函数
        x = self.drop(x)  # dropout层，丢弃一定比例的神经元
        x = self.fc2(x)  # 第二个全连接层
        x = self.drop(x)  # dropout层
        return x


"""
    搭建一个Transformer的Block
"""


class Block(nn.Module):
    ''' 这里的参数比较多，比较乱，注释写的也不一定清楚，还是需要自己写完整体代码后再回来多琢磨一下 '''

    def __init__(self,
                 dim,  # 每个token的维度
                 num_heads,  # 多头自注意的头数
                 mlp_ratio=4,  # MLP中，隐藏层维度是输入层的多少倍，默认4倍
                 qkv_bias=False,  # qkv偏置项
                 qk_scale=None,  # qk缩放因子
                 drop_ratio=0.,  # 多头自注意力机制最后的Linear后使用的dropout
                 attn_drop_ratio=0.,  # 生成qkv之后的dropout
                 drop_path_ratio=0.,  # drop_path的比例（在Encoder中，Multi-head后会drop_path）
                 act_layer=nn.GELU,  # 激活函数
                 norm_layer=nn.LayerNorm):  # 正则化层
        super().__init__()
        self.norm1 = norm_layer(dim)  # 对应于Encoder传入输入后遇到的第一个LayerNorm
        # 对应于Multi-Head Attention模块。之前我们已经实现了这个类，此处进行了类的实例化
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              atte_drop_ratio=attn_drop_ratio,
                              proj_drop_ratio=drop_ratio)
        # 如果drop_path_ratio>0则使用droppath，否则不做任何更改
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)  # 对应于第二个LayerNorm层
        # 定义mlp层
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_feature=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        '''
            按照Encoder所提供的特征图，逐个模块实现即可
            第一个残差连接内部的层（第一个+号前的部分）：LayerNorm --> Multi-Head Attention --> DropPath，另外还要做一个残差
            第二个+号前的部分，同理。
        '''
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4.0, qkv_bias=True, qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None, act_layer=None):
        '''
            num_classes=1000，即分类的数量共1000个，因为我们将来在ImageNet上做训练
            depth=12，即ViT的深度为12，也就是把Block循环12次，也就是把Transformer Encoder叠加12层
            mlp_ratio=4，即中间隐藏层的维度是输入维度的4倍
            representation_size - 在MLP Head处是否使用一个全连接层和一个激活函数（就是ViT那张示意图上面的MLP Head那里）【不过这个没太看懂】
            distilled - 蒸馏部分的DiT的一个参数
            norm_layer - 标准化层
            act_layer - 激活函数
        '''
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)  # 设置一个较小的参数eps防止除以0
        act_layer = act_layer or nn.GELU()
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches  # 得到patches的个数
        ''' 使用nn.Parameter构建可训练的参数，用零矩阵初始化
            参数[1, 1, embed_dim]的意思 - 第一个1是batch_size，后两个参数是1*768的意思（cls token本身是一个长度768的一维向量） '''
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        ''' 这个是与DiT相关的参数，可以暂时先不管那么多 '''
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        ''' pos_embed三个参数的含义 - 第一个1是batch_size，后两个参数是197 * 768 '''
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)
        ''' 构建dropout率，是一个列表，列表内容是一个等差数列
            三个参数的含义 - 首项、尾项、项数 '''
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        ''' 列表推导式构建
            使用nn.Sequential将列表中的所有模块打包成一个整体 '''
        self.block = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i], norm_layer=norm_layer,
                  act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)  # Transformer层后面的LayerNorm

        ''' representation_size 是ViT在ImageNet 1K上面做训练涉及到的一个参数 '''
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # 分类头
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # 权重初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # B C H W -> B num_patches embed_dim
        x = self.patch_embed(x)
        # 1, 1, 768 --> B, 1, 768
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # 在dim=1上进行concat拼接
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        x = self.block(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            # 分别通过head和head_dist进行预测
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            # 如果是训练模式且不是脚本模式
            if self.training and not torch.jit.is_scripting():
                return x, x_dist
        else:
            x = self.head(x)  # 最后的Linear全连接层
        return x


def _init_vit_weights(m):
    if isinstance(m, nn.Linear):  # 判断m是否是一个线性层
        nn.init.trunc_normal_(m.weight, std=.01)  # 初始化参数权重，标准差为0.01
        if m.bias is not None:  # 如果线性层存在偏置项
            nn.init.zeros_(m.bias)  # 则对线性层的偏置项进行初始化，初始化为0

    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")  # 对卷积层的权重做一个初始化，凯明初始化适用于卷积
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)  # 把归一化层的权重初始化为1


def vit_base_patch16_224(num_classes: int = 1000, pretrained=False):
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


