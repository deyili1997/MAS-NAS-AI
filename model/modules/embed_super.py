import torch.nn as nn
import torch.nn.functional as F


class TokenEmbedSuper(nn.Module):

    def __init__(
        self,
        vocab_size,
        type_vocab_size,
        max_adm,
        super_embed_dim,
        uniform_=None,
        non_linear='linear',
    ):
        super().__init__()

        self.super_embed_dim = super_embed_dim
        self.sample_embed_dim = None

        # [num_embeddings, embed_dim]
        self.code_embed = nn.Embedding(vocab_size, super_embed_dim)
        self.type_embed = nn.Embedding(type_vocab_size, super_embed_dim)
        self.adm_embed = nn.Embedding(max_adm + 2, super_embed_dim)  # +2 for [PAD] and [CLS]

        self.samples = {}
        self.profiling = False

        self._reset_parameters(uniform_=uniform_, non_linear=non_linear)

    def _reset_parameters(self, uniform_=None, non_linear='linear'):
        if uniform_ is None:
            nn.init.xavier_uniform_(self.code_embed.weight)
            nn.init.xavier_uniform_(self.type_embed.weight)
            nn.init.xavier_uniform_(self.adm_embed.weight)
        else:
            uniform_(self.code_embed.weight, non_linear=non_linear)
            uniform_(self.type_embed.weight, non_linear=non_linear)
            uniform_(self.adm_embed.weight, non_linear=non_linear)

    def profile(self, mode=True):
        self.profiling = mode

    def set_sample_config(self, sample_embed_dim):
        assert sample_embed_dim <= self.super_embed_dim, \
            f"sample_embed_dim ({sample_embed_dim}) cannot exceed super_embed_dim ({self.super_embed_dim})"
        self.sample_embed_dim = sample_embed_dim
        self._sample_parameters()

    def sample_parameters(self, resample=False):
        if self.sample_embed_dim is None:
            raise ValueError("sample_embed_dim is None. Call set_sample_config() before forward().")
        if self.profiling or resample or not self.samples:
            return self._sample_parameters()
        return self.samples

    def _sample_parameters(self):
        # Keep all vocab rows, only slice embedding dimension
        self.samples = {
            "code_embed": self.code_embed.weight[:, :self.sample_embed_dim],  # [vocab_size, sample_embed_dim]
            "type_embed": self.type_embed.weight[:, :self.sample_embed_dim],  # [type_vocab_size, sample_embed_dim]
            "adm_embed": self.adm_embed.weight[:, :self.sample_embed_dim],    # [max_adm+2, sample_embed_dim]
        }
        return self.samples

    def forward(self, input_ids, token_types, adm_index):
        """
        input_ids   : [B, L]
        token_types : [B, L]
        adm_index   : [B, L]

        return:
        embeddings  : [B, L, sample_embed_dim]
        """
        self.sample_parameters()

        code_x = F.embedding(input_ids, self.samples["code_embed"])
        type_x = F.embedding(token_types, self.samples["type_embed"])
        adm_x = F.embedding(adm_index, self.samples["adm_embed"])
        x = code_x + type_x + adm_x
        return x