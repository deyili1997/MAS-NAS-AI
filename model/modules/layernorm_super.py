import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNormSuper(torch.nn.LayerNorm):
    def __init__(self, super_embed_dim):
        super().__init__(super_embed_dim)
        """
            定义 LayerNormSuper 类，并继承 torch.nn.LayerNorm。
            继承后，该类天然拥有 LayerNorm 的参数：
            self.weight（也叫 gamma）：形状 [normalized_shape]
            self.bias（也叫 beta）：形状 [normalized_shape]
            self.eps：数值稳定用的小常数
            在这里 normalized_shape 就是 super_embed_dim。
        """

        # the largest embed dim
        self.super_embed_dim = super_embed_dim

        # the current sampled embed dim
        self.sample_embed_dim = None

        self.samples = {}
        self.profiling = False
        # profiling 开关：
	    # •	False：默认使用缓存 samples
	    # •	True：每次都重新切片（保证参数与当前 sample 配置一致，常用于复杂度统计或频繁切换配置时）

    def profile(self, mode=True):
        self.profiling = mode

    def sample_parameters(self, resample=False):
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def _sample_parameters(self):
        self.samples['weight'] = self.weight[:self.sample_embed_dim]
        self.samples['bias'] = self.bias[:self.sample_embed_dim]
        return self.samples

    def set_sample_config(self, sample_embed_dim):
        self.sample_embed_dim = sample_embed_dim
        self._sample_parameters()

    def forward(self, x):
        self.sample_parameters()
        return F.layer_norm(x, (self.sample_embed_dim,), weight=self.samples['weight'], bias=self.samples['bias'], eps=self.eps)

    def calc_sampled_param_num(self):
        assert 'weight' in self.samples.keys()
        assert 'bias' in self.samples.keys()
        return self.samples['weight'].numel() + self.samples['bias'].numel()

    def get_complexity(self, sequence_length):
        return sequence_length * self.sample_embed_dim
