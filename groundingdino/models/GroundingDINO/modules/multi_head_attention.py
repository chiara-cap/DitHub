# Description: Multi-head attention module from pytorch but with nn.Linear instead of torch.nn.functional.linear
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.functional import dropout, softmax

import warnings


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 in_features,
                 head_num,
                 bias=True,
                 activation=None,
                 dropout_p=0.0):
        """Multi-head attention.

        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttention, self).__init__()
        if in_features % head_num != 0:
            raise ValueError(f'`in_features`({in_features}) should be divisible by `head_num`({head_num})')
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.dropout_p = dropout_p

        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.linear_o = nn.Linear(in_features, in_features, bias)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.linear_q.weight)
        xavier_uniform_(self.linear_k.weight)
        xavier_uniform_(self.linear_v.weight)

        constant_(self.linear_q.bias, 0.)
        constant_(self.linear_k.bias, 0.)
        constant_(self.linear_v.bias, 0.)
        constant_(self.linear_o.bias, 0.)


    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None, **kwargs) -> Tuple[Tensor, Optional[Tensor]]:
        tgt_len, bsz, embed_dim = query.size()
        # q, k, v = self.linear_q(query), self.linear_k(key), self.linear_v(value)
        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)

        head_dim = self.in_features // self.head_num
        scaling = float(head_dim) ** -0.5
        q = q * scaling
        if attn_mask is not None:
            assert attn_mask.dtype in [
                torch.float32,torch.float64,torch.float16,torch.uint8,torch.bool,
            ], f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
            if attn_mask.dtype == torch.uint8:
                warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
                attn_mask = attn_mask.to(torch.bool)

            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError("The size of the 2D attn_mask is not correct.")
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [bsz * self.head_num, query.size(0), key.size(0)]:
                    raise RuntimeError("The size of the 3D attn_mask is not correct.")
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")


        q = q.contiguous().view(tgt_len, bsz * self.head_num, head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.head_num, head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.head_num, head_dim).transpose(0, 1)

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * self.head_num, tgt_len, src_len]

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, self.head_num, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            attn_output_weights = attn_output_weights.view(bsz * self.head_num, tgt_len, src_len)

        attn_output_weights = softmax(attn_output_weights, dim=-1)
        attn_output_weights = dropout(attn_output_weights, p=self.dropout_p, training=self.training)

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * self.head_num, tgt_len, head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.linear_o(attn_output)

        if not need_weights:
            return attn_output, None
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, self.head_num, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / self.head_num
