import torch.nn as nn
import torch.nn.functional as F
from model.modules.activation import gelu
from model.modules.drop_path import DropPath
from model.modules.layernorm_super import LayerNormSuper
from model.modules.linear_super import LinearSuper
from model.modules.attention_super import AttentionSuper

class TransformerEncoderLayer(nn.Module):
    """
        •	这是初始化函数，输入的是“超网（最大网络）”层配置：
        •	dim: 超网 embedding 维度（最大通道数）
        •	num_heads: 超网 head 数
        •	mlp_ratio: FFN 隐藏层维度比例（通常 4）
        •	dropout: FFN/投影的 dropout
        •	attn_drop: attention 权重 dropout
        •	drop_path: stochastic depth 概率
        •	pre_norm: True 表示 Pre-LN（LayerNorm 在子层前）
        •	scale: 是否对不同子网规模做输出幅度补偿
        •	change_qkv: 是否允许 QKV 维度随子网变化
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, dropout=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, pre_norm=True, scale=False, change_qkv=False):
        super().__init__()
        # __init__：建立该层encoder“超网最大结构”，并准备子网切片接口
        # the configs of super arch of the encoder, three dimension [embed_dim, mlp_ratio, and num_heads]
        self.super_embed_dim = dim # 后续 LayerNormSuper / LinearSuper 会以此为最大维度创建参数。
        self.super_mlp_ratio = mlp_ratio # 保存超网 MLP ratio，用于后续可选的 scale 补偿（在 FFN 输出处）。
        self.super_ffn_embed_dim_this_layer = int(mlp_ratio * dim) # 计算超网 FFN 中间隐藏维度,这对应 Transformer FFN 的第一层 Linear 输出维度
        self.super_num_heads = num_heads # 保存超网 head 数
        
        self.normalize_before = pre_norm # normalize_before 控制 LN 放在子层前还是后（Pre-LN vs Post-LN）
        self.super_dropout = attn_drop # attention 权重（softmax 后的注意力矩阵）上的 dropout 概率
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity() # drop_path 是 stochastic depth：按概率把整条残差分支置零
        # （注意，是在样本级别上，每个样本一个开关，要么整条残差分支保留，并且输出除以 keep_prob 以保持期望一致，要么整条残差分支全丢）
        self.scale = scale # 决定是否对“不同子网规模带来的输出幅度差异”做数值缩放（rescale）补偿
        
        # 这些在每次采样子网时都会被写入。forward 依赖它们，所以必须先调用 set_sample_config(...)
        self.sample_embed_dim = None # 子网 embedding 维度（输入 x 的最后一维）
        self.sample_mlp_ratio = None # 子网 FFN ratio
        self.sample_ffn_embed_dim_this_layer = None # 子网 FFN 隐藏维度 int(sample_embed_dim * sample_mlp_ratio)
        self.sample_num_heads_this_layer = None # 子网 head 数
        self.sample_scale = None #  这里预留但未在 forward 直接使用（attention 内部用 sample_scale）
        self.sample_dropout = None # 子网 FFN dropout
        self.sample_attn_dropout = None # 子网 attention dropout
        
        
        self.is_identity_layer = None # 用于 NAS / 动态深度：某些层可被裁掉。
        self.attn = AttentionSuper(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=dropout, scale=self.scale, change_qkv=change_qkv
            ) # forward return [B, N, sample_qk_embed_dim]
        
        self.attn_layer_norm = LayerNormSuper(self.super_embed_dim) # attn_layer_norm 用于 attention 子层，参数最大维度都是 super_embed_dim，运行时切成 sample_embed_dim。
        self.ffn_layer_norm = LayerNormSuper(self.super_embed_dim) # ffn_layer_norm 用于 FFN 子层，参数最大维度都是 super_embed_dim，运行时切成 sample_embed_dim。
        self.activation_fn = gelu

        self.fc1 = LinearSuper(super_in_dim=self.super_embed_dim, super_out_dim=self.super_ffn_embed_dim_this_layer)
        self.fc2 = LinearSuper(super_in_dim=self.super_ffn_embed_dim_this_layer, super_out_dim=self.super_embed_dim)
    
    
    def set_sample_config(self, is_identity_layer, sample_embed_dim=None, 
                          sample_mlp_ratio=None, sample_num_heads=None, 
                          sample_dropout=None, sample_attn_dropout=None, 
                          sample_out_dim=None):
        
        # 把“当前子网结构”写入并下发到子模块
        # 每次采样一个子网时调用，配置本层：
        # •	是否跳过该层（identity）
        # •	子网 dim、ratio、heads
        # •	dropout 设置
        # •	sample_out_dim：本层输出维度（可能用于 stage 过渡/下采样等）
        
        if is_identity_layer:
            self.is_identity_layer = True
            return

        self.is_identity_layer = False
        
        self.sample_embed_dim = sample_embed_dim
        self.sample_out_dim = sample_out_dim
        self.sample_mlp_ratio = sample_mlp_ratio
        self.sample_ffn_embed_dim_this_layer = int(sample_embed_dim*sample_mlp_ratio)
        self.sample_num_heads_this_layer = sample_num_heads
        
        self.sample_dropout = sample_dropout
        self.sample_attn_dropout = sample_attn_dropout
        # 下发 LN 子网维度
        self.attn_layer_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim) 
        
        # 给 attention 模块下发子网配置：
        # •	sample_in_embed_dim = sample_embed_dim（输入维度）
        # •	sample_num_heads = sample_num_heads_this_layer（head 数）
        # •	sample_q_embed_dim = sample_num_heads_this_layer * 64
        # •	这行隐含一个强假设：每个 head 固定 64 维，所以总 QK 维度 = H*64
        # •	因此 D=64，不随子网变动；这在某些 ViT/AutoFormer 设计里是刻意的（便于不同 head 数组合）
        self.attn.set_sample_config(sample_q_embed_dim=self.sample_num_heads_this_layer*64, 
                                    sample_num_heads=self.sample_num_heads_this_layer, 
                                    sample_in_embed_dim=self.sample_embed_dim)
        
        # FFN 第一层裁剪为：
	    # •	sample_embed_dim -> sample_ffn_embed_dim_this_layer
        self.fc1.set_sample_config(sample_in_dim=self.sample_embed_dim, 
                                   sample_out_dim=self.sample_ffn_embed_dim_this_layer)
        # FFN 第二层裁剪为：
	    # •	sample_ffn_embed_dim_this_layer -> sample_out_dim
        # 注意：这里输出不是写死 sample_embed_dim，而是用 sample_out_dim
        self.fc2.set_sample_config(sample_in_dim=self.sample_ffn_embed_dim_this_layer, 
                                   sample_out_dim=self.sample_out_dim)
        
        self.ffn_layer_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim)
        
    def forward(self, x, attn_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(B, L, D)`

        Returns:
            encoded output of shape `(B, L, D)`
            
        1. Attention子层：
            x <- x + DropPath(Dropout(Attn(LayerNorm(x))))  
        2. Feed-Forward子层：
            x <- x + DropPath(Dropout(FC2(Dropout(sigma((FC1(LayerNorm(x))))))  
            """
        # 若该层被采样为 identity，直接返回输入，不做 attention/FFN。
        if self.is_identity_layer:
            return x
        residual = x # 保存残差分支的输入，用于后面 residual + x
        x = self.maybe_layer_norm(self.attn_layer_norm, x, before=True)
        x = self.attn(x, attn_mask) # [B, L, D]
        x = F.dropout(x, p=self.sample_attn_dropout, training=self.training)
         # •以概率 drop_path 把整条 residual 分支置 0（样本级)否则按 keep_prob 做缩放以保持期望。
        x = self.drop_path(x) 
        x = residual + x # 残差连接：把 attention 分支输出加回原输入。
        x = self.maybe_layer_norm(self.attn_layer_norm, x, after=True)
        # print("attn :", time.time() - start_time)
        # compute the ffn
        # start_time = time.time()
        residual = x # 保存残差分支的输入，用于后面 residual + x
        x = self.maybe_layer_norm(self.ffn_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        if self.scale:
            x = x * (self.super_mlp_ratio / self.sample_mlp_ratio)
        x = self.drop_path(x)
        x = residual + x # 残差连接：把 FFN 分支输出加回原输入。
        x = self.maybe_layer_norm(self.ffn_layer_norm, x, after=True)
        # print("ffn :", time.time() - start_time)
        return x

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x