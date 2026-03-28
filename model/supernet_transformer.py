import torch
import torch.nn as nn
import os
import sys
import torch.nn.functional as F

# Add project root (.../MAS-NAS) to import path
sys.path.append(os.path.abspath("../.."))
sys.executable

from utils.para_init import trunc_normal_
from model.modules.embed_super import TokenEmbedSuper
from model.modules.encoder_super import TransformerEncoderLayer
from model.modules.layernorm_super import LayerNormSuper
from model.modules.linear_super import LinearSuper
from model.modules.activation import gelu


class TransformerSuper(nn.Module):
    def __init__(
        self,
        num_classes: int,
        vocab_size: int,                  # 新增：MLM输出维度
        embed_dim: int,
        mlp_ratio: float,
        depth: int,
        num_heads: int,
        qkv_bias: bool = False,
        qk_scale=None,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        pre_norm: bool = True,
        scale: bool = False,
        change_qkv: bool = False,
        type_vocab_size: int = 7,
        max_adm_num: int = 8,
    ):
        super().__init__()

        # super arch configs
        self.super_embed_dim = embed_dim
        self.super_mlp_ratio = mlp_ratio
        self.super_layer_num = depth
        self.super_num_heads = num_heads
        self.super_dropout = drop_rate
        self.super_attn_dropout = attn_drop_rate
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.pre_norm = pre_norm
        self.scale = scale

        # token embedding
        self.token_embed = TokenEmbedSuper(vocab_size, type_vocab_size, max_adm_num, embed_dim)

        # sampled subnet configs
        self.sample_embed_dim = None
        self.sample_mlp_ratio = None
        self.sample_layer_num = None
        self.sample_num_heads = None
        self.sample_dropout = None
        self.sample_output_dim = None

        # transformer blocks
        self.blocks = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        for i in range(depth):
            self.blocks.append(
                TransformerEncoderLayer(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    dropout=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    pre_norm=pre_norm,
                    scale=self.scale,
                    change_qkv=change_qkv,
                )
            )

        if self.pre_norm:
            self.norm = LayerNormSuper(super_embed_dim=embed_dim)

        # classification head: CLS -> class logits
        self.head = LinearSuper(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # MLM head: hidden state -> vocab logits
        # 这里做成 BERT 风格的小 head
        self.mlm_dense = LinearSuper(embed_dim, embed_dim)
        self.mlm_ln = LayerNormSuper(super_embed_dim=embed_dim)
        self.mlm_head = LinearSuper(embed_dim, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        return self.head

    def get_mlm_head(self):
        return self.mlm_head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = LinearSuper(self.super_embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.sample_embed_dim is not None:
            self.head.set_sample_config(self.sample_embed_dim[-1], self.num_classes)

    def set_sample_config(self, config: dict):
        self.sample_embed_dim = config['embed_dim']
        self.sample_mlp_ratio = config['mlp_ratio']
        self.sample_layer_num = config['layer_num']
        self.sample_num_heads = config['num_heads']

        # Validate config lengths and values early to avoid runtime shape errors.
        if not (len(self.sample_embed_dim) == len(self.sample_mlp_ratio) == len(self.sample_num_heads)):
            raise ValueError(
                "subnet config length mismatch: embed_dim, mlp_ratio, and num_heads must have the same length"
            )
        if self.sample_layer_num < 1 or self.sample_layer_num > len(self.sample_embed_dim):
            raise ValueError(
                f"layer_num must be in [1, {len(self.sample_embed_dim)}], got {self.sample_layer_num}"
            )

        active_embed_dims = self.sample_embed_dim[:self.sample_layer_num]
        if any(d > self.super_embed_dim for d in active_embed_dims):
            raise ValueError(
                f"sample embed_dim cannot exceed super embed_dim ({self.super_embed_dim}): {active_embed_dims}"
            )

        # Current encoder keeps residual add in the same hidden width inside each layer.
        # Enforce fixed embed dim across active layers until transition projections are implemented.
        if len(set(active_embed_dims)) != 1:
            raise ValueError(
                "Current MAS-NAS encoder requires constant embed_dim across active layers. "
                f"Got embed_dim={active_embed_dims}."
            )

        for i, (d, h) in enumerate(zip(self.sample_embed_dim[:self.sample_layer_num], self.sample_num_heads[:self.sample_layer_num])):
            if d % h != 0:
                raise ValueError(
                    f"embed_dim must be divisible by num_heads at layer {i}: embed_dim={d}, num_heads={h}"
                )

        self.sample_dropout = calc_dropout(
            self.super_dropout,
            self.sample_embed_dim[0],
            self.super_embed_dim
        )
        self.token_embed.set_sample_config(self.sample_embed_dim[0])

        self.sample_output_dim = [out_dim for out_dim in self.sample_embed_dim[1:]] + [self.sample_embed_dim[-1]]

        for i, block in enumerate(self.blocks):
            if i < self.sample_layer_num:
                sample_dropout = calc_dropout(
                    self.super_dropout,
                    self.sample_embed_dim[i],
                    self.super_embed_dim
                )
                sample_attn_dropout = calc_dropout(
                    self.super_attn_dropout,
                    self.sample_embed_dim[i],
                    self.super_embed_dim
                )
                block.set_sample_config(
                    is_identity_layer=False,
                    sample_embed_dim=self.sample_embed_dim[i],
                    sample_mlp_ratio=self.sample_mlp_ratio[i],
                    sample_num_heads=self.sample_num_heads[i],
                    sample_dropout=sample_dropout,
                    sample_out_dim=self.sample_output_dim[i],
                    sample_attn_dropout=sample_attn_dropout
                )
            else:
                block.set_sample_config(is_identity_layer=True)

        if self.pre_norm:
            self.norm.set_sample_config(self.sample_embed_dim[-1])

        if self.num_classes > 0:
            self.head.set_sample_config(self.sample_embed_dim[-1], self.num_classes)

        # add this block
        mlm_dim = self.sample_embed_dim[-1]
        self.mlm_dense.set_sample_config(mlm_dim, mlm_dim)
        self.mlm_ln.set_sample_config(mlm_dim)
        self.mlm_head.set_sample_config(mlm_dim, self.vocab_size)

    def forward_features(self, input_ids, token_types, adm_index, attn_mask):
        """
        x: [B, L] token ids
        return:
            hidden_states: [B, L, D]
        """
        x = self.token_embed(input_ids, token_types, adm_index)  # [B, L, D]
        x = F.dropout(x, p=self.sample_dropout, training=self.training)

        for blk in self.blocks:
            x = blk(x, attn_mask)

        if self.pre_norm:
            x = self.norm(x)

        return x  # [B, L, D]

    def forward(self, input_ids, token_types, adm_index, attn_mask, task="mlm"):
        hidden_states = self.forward_features(input_ids, token_types, adm_index, attn_mask)

        if task == "cls":
            return self.head(hidden_states[:, 0])

        if task == "mlm":
            return self.mlm_head(self.mlm_ln(gelu(self.mlm_dense(hidden_states))))

        raise ValueError(f"Unsupported task: {task}")


def calc_dropout(dropout, sample_embed_dim, super_embed_dim):
    return dropout * 1.0 * sample_embed_dim / super_embed_dim