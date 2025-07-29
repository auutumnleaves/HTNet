'''
参考链接：https://zhuanlan.zhihu.com/p/694913284
'''

import torch
import torch.nn.functional as F

# input是一个张量x，其形状为(batch_size, seq_len, feature_dim)
x = torch.randn(2, 3, 4)  # 形状(batch_size, seq_len, feature_dim)
print("原始输入：", x)
print("-----")

# 定义头数和每个头的维度
num_heads = 2
head_dim = 2
# feature_dim必须等于num_heads*head_dim
assert x.size(-1) == num_heads * head_dim

# 定义线性层用于将x转为Q,K,V矩阵
linear_q = torch.nn.Linear(4, 4)
linear_k = torch.nn.Linear(4, 4)
linear_v = torch.nn.Linear(4, 4)
# 调用上述定义的线性层，获取Q,K,V
Q = linear_q(x)  # 形状(batch_size, seq_len, feature_dim)
K = linear_k(x)  # 形状(batch_size, seq_len, feature_dim)
V = linear_v(x)  # 形状(batch_size, seq_len, feature_dim)

# 由于是多头注意力，因此需要把完整的Q,K,V平均分到每个头上
def split_heads(tensor, num_heads):
    batch_size, seq_len, feature_dim = tensor.size()
    ''' //是地板除，能使除法的结果是int型，如果使用/，即使能整除，结果也是float。事实上，对于任何计算维度值的地方，都应该使用// '''
    head_dim = feature_dim // num_heads
    ''' transpose(1,2)的意思是将下标1和2的维度交换一下，即形状转为(batch_size, num_heads, seq_len, head_dim)
        之所以要交换维度，是因为seq_len和head_dim相邻，才有利于进行后续的运算等操作，而seq_len, num_heads, head_dim这种排列方式是没意义的
        类比一下，就好比RGB图像的形状应该是(H, W, 3)，而不应该是(H, 3, W)。
        而之所以不直接view(batch_size, num_heads, seq_len, head_dim)，是因为我们要遵守内存的连续存储，不能随意分割'''
    output = tensor.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    return output  # 形状(batch_size, num_heads, seq_len, head_dim)
Q = split_heads(Q, num_heads)  # 形状(batch_size, num_heads, seq_len, head_dim)
K = split_heads(K, num_heads)  # 形状(batch_size, num_heads, seq_len, head_dim)
V = split_heads(V, num_heads)  # 形状(batch_size, num_heads, seq_len, head_dim)

# 计算Q和K的点积，作为相似度分数
raw_weights = torch.matmul(Q, K.transpose(-2, -1))  # 形状(batch_size, num_heads, seq_len, seq_len)

# 对自注意力原始权重进行缩放
''' K.size(-1)就是head_dim
    之所以Q和K相乘之后还要缩放，是因为，Q和K的元素相乘之后，所得到结果的方差会增大，从而导致梯度爆炸等问题，
    经过缩放，可以使方差恢复到原来的。（）'''
scale_factor = K.size(-1) ** 0.5
scaled_weights = raw_weights / scale_factor  # 形状(batch_size, num_heads, seq_len, seq_len)
# 对缩放后的权重进行softmax归一化，得到注意力权重
''' Q和K矩阵相乘的结果，经过缩放后，还要与V矩阵作加权求和，其中的权重就是利用softmax层得来的
    dim=-1表示在最后一个维度上进行softmax，而scaled_weights最后一个维度是seq_len
    注意，有两个seq_len维度，对后一个seq_len分配权重、且权重之和为1，而前一个seq_len代表了每个Token
    即，对于各Token，都为它与其他每个Token的关系分配一个权重，且权重之和为1'''
attn_weights = F.softmax(scaled_weights, dim=-1)  # 形状(batch_size, num_heads, seq_len, seq_len)

# 将注意力权重应用于V向量，计算加权和，得到加权信息
attn_outputs = torch.matmul(attn_weights, V)  # 形状(batch_size, num_heads, seq_len, head_dim)

# 将所有头的结果拼接起来
def combine_heads(tensor, num_heads):
    batch_size, num_heads, seq_len, head_dim = tensor.size()
    feature_dim = num_heads * head_dim
    output = tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, feature_dim)
    return output  # 形状(batch_size, seq_len, feature_dim)
attn_outputs = combine_heads(attn_outputs, num_heads)  # 形状(batch_size, seq_len, feature_dim)

# 对拼接后的结果作线性变换
linear_out = torch.nn.Linear(4, 4)
attn_outputs = linear_out(attn_outputs)  # 形状(batch_size, seq_len, feature_dim)
print("加权信息：", attn_outputs)