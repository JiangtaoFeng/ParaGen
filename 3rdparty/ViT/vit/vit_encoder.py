from torch import Tensor
import torch
import torch.nn as nn

from paragen.modules.encoders import AbstractEncoder, register_encoder
from paragen.modules.encoders.layers.transformer_encoder_layer import TransformerEncoderLayer


@register_encoder
class ImageTransformerEncoder(AbstractEncoder):
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
        # self._layers = nn.ModuleList([TransformerEncoderLayer(d_model=self._d_model,
        #                                                       nhead=self._n_head,
        #                                                       dim_feedforward=self._dim_feedforward,
        #                                                       dropout=self._dropout,
        #                                                       attention_dropout=self._attention_dropout,
        #                                                       activation=self._activation,
        #                                                       normalize_before=self._normalize_before)
        #                               for _ in range(self._num_layers)])
        # self._norm = nn.LayerNorm(self._d_model) if self._normalize_before else None

    def _forward(self, x: Tensor):
        r"""
        Args:
            x: tokens in src side.
              :math:`(N, H, W, 3)` where N is the batch size, H is the image height, W is the image width and 3 is the rgb channel.

        Outputs:
            - source token hidden representation.
              :math:`(S, N, E)` where S is the source sequence length, N is the batch size,
              E is the embedding size.
        """
        # x = x.transpose(0, 1)
        # for layer in self._layers:
        #     x = layer(x)
        #
        # if self._norm is not None:
        #     x = self._norm(x)
        return x.mean(dim=1)

    @property
    def d_model(self):
        return self._d_model

    @property
    def out_dim(self):
        return self._d_model
