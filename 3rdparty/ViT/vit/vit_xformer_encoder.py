from typing import Optional
import importlib

from torch import Tensor
import torch.nn as nn

from paragen.modules.encoders import AbstractEncoder, register_encoder
from paragen.modules.encoders.layers import AbstractEncoderLayer
from paragen.modules.layers.feed_forward import FFN


@register_encoder
class ImageXformerEncoder(AbstractEncoder):
    """
    TransformerEncoder is a transformer encoder.

    Args:
        num_layers: number of encoder layers
        d_model: feature dimension
        n_head: head numbers of multihead attention
        dim_feedforward: dimensionality of inner vector space
        dropout: dropout rate
        activation: activation function used in feed-forward network
        return_seed: return with sequence representation
        normalize_before: use pre-norm fashion, default as post-norm.
            Pre-norm suit deep nets while post-norm achieve better results when nets are shallow.
        name: module name
    """

    def __init__(self,
                 num_layers,
                 xformer_type='MultiheadAttention',
                 d_model=512,
                 n_head=8,
                 dim_feedforward=2048,
                 dropout=0.1,
                 attention_dropout=0.,
                 activation='relu',
                 return_seed=True,
                 normalize_before=False,
                 embed_layer_norm=False,
                 max_pos=1024,
                 name=None):
        super().__init__()
        self._num_layers = num_layers
        self._xformer_type = xformer_type
        self._d_model = d_model
        self._n_head = n_head
        self._dim_feedforward = dim_feedforward
        self._dropout = dropout
        self._attention_dropout = attention_dropout
        self._activation = activation
        self._return_seed = return_seed
        self._normalize_before = normalize_before
        self._name = name
        self._embed_layer_norm = embed_layer_norm
        self._max_pos = max_pos

        self._special_tokens = None
        self._embed, self._h_pos_embed, self._w_pos_embed = None, None, None
        self._embed_norm, self._embed_dropout = None, None
        self._cls_token = None
        self._layers = None
        self._norm = None

    def build(self):
        """
        Build computational modules.
        """
        self._layers = nn.ModuleList([XformerEncoderLayer(d_model=self._d_model,
                                                          nhead=self._n_head,
                                                          xformer_type=self._xformer_type,
                                                          dim_feedforward=self._dim_feedforward,
                                                          dropout=self._dropout,
                                                          attention_dropout=self._attention_dropout,
                                                          activation=self._activation,
                                                          normalize_before=self._normalize_before)
                                      for _ in range(self._num_layers)])
        self._norm = nn.LayerNorm(self._d_model) if self._normalize_before else None

    def forward(self, x: Tensor):
        r"""
        Args:
            x: tokens in src side.
              :math:`(N, H, W, 3)` where N is the batch size, H is the image height, W is the image width and 3 is the rgb channel.

        Outputs:
            - source token hidden representation.
              :math:`(S, N, E)` where S is the source sequence length, N is the batch size,
              E is the embedding size.
        """
        x = x.transpose(0, 1)
        for layer in self._layers:
            x = layer(x)

        if self._norm is not None:
            x = self._norm(x)

        return x[0]

    @property
    def d_model(self):
        return self._d_model

    @property
    def out_dim(self):
        return self._d_model


class XformerEncoderLayer(AbstractEncoderLayer):
    """
    TransformerEncoderLayer performs one layer of transformer operation, namely self-attention and feed-forward network.

    Args:
        d_model: feature dimension
        nhead: head numbers of multihead attention
        dim_feedforward: dimensionality of inner vector space
        dropout: dropout rate
        activation: activation function used in feed-forward network
        normalize_before: use pre-norm fashion, default as post-norm.
            Pre-norm suit deep nets while post-norm achieve better results when nets are shallow.
    """

    def __init__(self,
                 d_model,
                 nhead,
                 xformer_type='MultiheadAttention',
                 dim_feedforward=2048,
                 dropout=0.1,
                 attention_dropout=0.,
                 activation="relu",
                 normalize_before=False,):
        super(XformerEncoderLayer, self).__init__()
        self.normalize_before = normalize_before
        mod = importlib.import_module('efficient_attention')
        cls = getattr(mod, xformer_type)
        self.self_attn = cls(d_model, nhead, dropout=attention_dropout)
        # Implementation of Feedforward model
        self.ffn = FFN(d_model, dim_feedforward=dim_feedforward, activation=activation)

        self.self_attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,
                src: Tensor,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
                :math:`(S, B, D)`, where S is sequence length, B is batch size and D is feature dimension
            src_mask: the attention mask for the src sequence (optional).
                :math:`(S, S)`, where S is sequence length.
            src_key_padding_mask: the mask for the src keys per batch (optional).
                :math: `(B, S)`, where B is batch size and S is sequence length
        """
        residual = src
        if self.normalize_before:
            src = self.self_attn_norm(src)
        src = self.self_attn(src, src, src, attn_mask=src_mask,
                             key_padding_mask=src_key_padding_mask)[0]
        src = self.dropout1(src)
        src = residual + src
        if not self.normalize_before:
            src = self.self_attn_norm(src)

        residual = src
        if self.normalize_before:
            src = self.ffn_norm(src)
        src = self.ffn(src)
        src = self.dropout2(src)
        src = residual + src
        if not self.normalize_before:
            src = self.ffn_norm(src)
        return src

