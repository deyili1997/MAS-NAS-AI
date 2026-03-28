import torch
import torch.nn as nn

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training: # training：是否处于训练模式；推理时一般不丢弃。
        return x
    keep_prob = 1 - drop_prob # keep_prob 是保留概率，即不丢弃的概率。
    # •	构造随机 mask 的形状 shape。
    # •	x.shape[0] 是 batch 维 B。
    # •	(1,) * (x.ndim - 1) 会生成若干个 1，使 mask 在除 batch 外的所有维度都是 1。
    # 这行的关键含义：
    # mask 形状是 (B, 1, 1, …, 1)，这样会在后续广播到 x 的所有非 batch 维度上。
    # 举例：
    # •	如果 x 是 (B, N, D)（Transformer 常见），则 x.ndim=3，shape = (B, 1, 1)
    # •	如果 x 是 (B, C, H, W)（CNN 常见），则 shape = (B, 1, 1, 1)
    # 因此它实现的是：每个样本一个开关，要么整条残差分支保留，要么整条残差分支全丢。
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    # •	生成 shape 大小的均匀随机数 U ~ Uniform(0,1)。
	# •	然后做 keep_prob + U，得到范围在 [keep_prob, keep_prob+1) 的随机张量。
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device) 
    # 这一步是为了下一行 floor_() 方便地产生 0/1 mask：
	# •	若 U < drop_prob，则 keep_prob + U < keep_prob + drop_prob = 1，floor 后为 0（丢弃）
	# •	若 U >= drop_prob，则 keep_prob + U >= 1，floor 后为 1（保留）
    # 也就是说最终保留概率正好是 keep_prob。
    random_tensor.floor_()  # binarize，并且它是每个样本一个标量（广播后作用于整条分支）。
    output = x.div(keep_prob) * random_tensor # 为什么要除以 keep_prob？为了让训练期的期望值与推理期对齐（inverted scaling）
    return output # 结果：要么是 x / keep_prob（保留），要么是 0（丢弃）。注意这里的 x 是残差分支的输出，而不是输入。

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)