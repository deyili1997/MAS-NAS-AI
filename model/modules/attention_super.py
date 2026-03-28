import torch
import torch.nn as nn
from model.modules.linear_super import LinearSuper
import numpy as np
import torch.nn.functional as F

class AttentionSuper(nn.Module):
    def __init__(self, super_embed_dim, num_heads=8, qkv_bias=False, qk_scale=None, 
                 attn_drop=0., proj_drop=0., scale=False, change_qkv = False):
        super().__init__()# 建立“超网注意力”的最大参数
        """
        构造函数参数：
        •	super_embed_dim：超网最大 embedding 维度（最大 C）
        •	num_heads：初始化时默认 head 数（超网默认值；运行时会用 sample_num_heads）
        •	qkv_bias：QKV 线性层是否带 bias
        •	qk_scale：注意力缩放因子（可手动指定）
        •	attn_drop / proj_drop：attention 权重 dropout、输出投影 dropout
        •	scale：是否对输出做额外宽度缩放补偿（工程技巧）
        •	change_qkv：是否允许 QKV 的输出维度随子网变化（核心开关之一）
        """

        self.num_heads = num_heads # 超网默认 heads（初始化时的 heads，仅用于初始化/默认值；真正运行用 sample_num_heads）
        head_dim = super_embed_dim // num_heads # 超网默认每头维度（标准 ViT 的设定）
        self.scale = qk_scale or head_dim ** -0.5 # 注意力缩放因子，默认是 head_dim 的倒数平方根（标准 ViT 的设定）
        self.super_embed_dim = super_embed_dim # 记录最大维度
        
        self.fc_scale = scale
        self.change_qkv = change_qkv
        if change_qkv:
            self.qkv = qkv_super(super_embed_dim, 3 * super_embed_dim, bias=qkv_bias)
        else:
            self.qkv = LinearSuper(super_embed_dim, 3 * super_embed_dim, bias=qkv_bias)
            
        # 这些是运行时子网配置：
        self.sample_qk_embed_dim = None # Q/K/V 投影后的总维度（通常 = 子网某个宽度）
        self.sample_num_heads = None # 子网的注意力头数
        self.sample_scale = None # 子网的 1/sqrt(head_dim)（注意力缩放）
        self.sample_in_embed_dim = None # 输入 x 的 embedding 维度（C）
        
        # 输出投影层（attention 输出再映射回 embedding 空间）：
        # 超网参数最大是 super_embed_dim -> super_embed_dim
        # 运行时会裁剪成 sample_qk_embed_dim -> sample_in_embed_dim（见 set_sample_config）
        self.proj = LinearSuper(super_embed_dim, super_embed_dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def set_sample_config(self, sample_q_embed_dim=None, sample_num_heads=None, sample_in_embed_dim=None):
        """
        •	设置本轮 forward 要用的子网结构参数：
        •	sample_in_embed_dim：输入维度
        •	sample_num_heads：head 数
        •	sample_q_embed_dim：当 change_qkv=True 时，QKV 的内部维度（总维度）
        """
        self.sample_in_embed_dim = sample_in_embed_dim
        self.sample_num_heads = sample_num_heads
        if not self.change_qkv:
            self.sample_qk_embed_dim = self.super_embed_dim
            self.sample_scale = (sample_in_embed_dim // self.sample_num_heads) ** -0.5 # 但缩放因子用子网的 head_dim 来算
        else:
            self.sample_qk_embed_dim = sample_q_embed_dim
            self.sample_scale = (self.sample_qk_embed_dim // self.sample_num_heads) ** -0.5
        # 将子网配置下发给 QKV 线性层
        self.qkv.set_sample_config(sample_in_dim=sample_in_embed_dim, sample_out_dim=3*self.sample_qk_embed_dim) 
        self.proj.set_sample_config(sample_in_dim=self.sample_qk_embed_dim, sample_out_dim=sample_in_embed_dim)

            
    def forward(self, x, attn_mask=None):
        # x: [B, L, D]
        # attn_mask: [B, L], 1 = valid token, 0 = padding

        B, L, D = x.shape

        qkv = self.qkv(x).reshape(B, L, 3, self.sample_num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v: [B, H, L, Dh]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # attention scores: [B, H, L, L]
        attn = (q @ k.transpose(-2, -1)) * self.sample_scale

        # 在 softmax 前应用 attention mask
        if attn_mask is not None:
            # attn_mask: [B, L]
            # -> [B, 1, 1, L]
            # 其中 1 表示保留，0 表示mask掉
            attn_mask = attn_mask[:, None, None, :]
            attn = attn.masked_fill(attn_mask == 0, float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)  # [B, H, L, L]

        x = (attn @ v).transpose(1, 2).reshape(B, L, -1)

        if self.fc_scale:
            x = x * (self.super_embed_dim / self.sample_qk_embed_dim)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class qkv_super(nn.Linear):
    # •	因为继承 nn.Linear，所以它天然就拥有：
    # •	self.weight：形状 [out_features, in_features]
    # •	self.bias：形状 [out_features] 或 None
    def __init__(self, super_in_dim, super_out_dim, bias=True, uniform_=None, non_linear='linear', scale=False):
    # •	调用父类 nn.Linear 的构造函数：真正创建超网参数张量：
	# •	self.weight 形状：[super_out_dim, super_in_dim]
	# •	若 bias=True，self.bias 形状：[super_out_dim]
	# •	这一步之后，这个模块本质上就是一个“最大尺寸”的 Linear。
        super().__init__(super_in_dim, super_out_dim, bias=bias)

    	# •	把最大输入/输出维度保存下来，作为“超网的上界配置”。
        # super_in_dim and super_out_dim indicate the largest network!
        self.super_in_dim = super_in_dim #  超网线性层的最大输入维度（最大 embedding dim）
        self.super_out_dim = super_out_dim #  超网线性层的最大输出维度（最大 embedding dim）

        # input_dim and output_dim indicate the current sampled size
        self.sample_in_dim = None # 这两个变量表示当前采样子网要用的输入/输出维度。
        self.sample_out_dim = None # 这两个变量表示当前采样子网要用的输入/输出维度。

        self.samples = {} # 一个字典缓存，用来保存“抽取出来的子网参数”：self.samples['weight']：
        # 子网权重 self.samples['bias']：子网 bias（或 None）

        self.scale = scale # 是否在 forward 后乘一个缩放因子（后面看到）
        self.profiling = False

    def profile(self, mode=True):
        self.profiling = mode

    def sample_parameters(self, resample=False):
        # if 必须刷新:
        #  重新生成子网参数
        # else:
        #  使用之前生成的子网参数
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def set_sample_config(self, sample_in_dim, sample_out_dim):
        # 设置子网维度并立即抽取参数
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim

        self._sample_parameters()

    def _sample_parameters(self):
        self.samples['weight'] = sample_weight(self.weight, self.sample_in_dim, self.sample_out_dim)
        self.samples['bias'] = self.bias
        self.sample_scale = self.super_out_dim/self.sample_out_dim
        if self.bias is not None:
            self.samples['bias'] = sample_bias(self.bias, self.sample_out_dim)
        return self.samples

    def forward(self, x):
        # x: [B, N, C]
        self.sample_parameters()
        return F.linear(x, self.samples['weight'], self.samples['bias']) * (self.sample_scale if self.scale else 1)


def sample_weight(weight, sample_in_dim, sample_out_dim):
    # weight: [super_out_dim, super_in_dim]
    sample_weight = weight[:, :sample_in_dim] # 子网只使用前 sample_in_dim 个 embedding 维度。
    # 所有 Q 行在索引 0,3,6,9,…
    # 所有 K 行在索引 1,4,7,10,…
    # 所有 V 行在索引 2,5,8,11,…
    sample_weight = torch.cat([sample_weight[i:sample_out_dim:3, :] for i in range(3)], dim =0) # [sample_out_dim, sample_in_dim]
    # 非常重要：顺序发生了改变
    # 先按交错方式分离 Q/K/V
	# 再按 block 形式拼接
    return sample_weight

def sample_bias(bias, sample_out_dim):
    sample_bias = bias[:sample_out_dim]

    return sample_bias