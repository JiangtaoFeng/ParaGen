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
                 height,
                 width,
                 patch_size,
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
        self._height, self._width = height, width
        self._patch_size = patch_size
        assert self._height % self._patch_size == 0 and self._width % self._patch_size == 0
        self._num_patch_h, self._num_patch_w = self._height // self._patch_size, self._width // self._patch_size
        self._num_patch = self._num_patch_h * self._num_patch_w
        self._patch_dim = self._patch_size ** 2 * 3
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
        self._embed, self._h_pos_embed, self._w_pos_embed, self._embed_norm, self._embed_dropout, self._norm = None, None, None, None, None, None
        self._cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self._layer, self._layers = None, None

    def build(self):
        """
        Build computational modules.
        """
        self._embed = nn.Linear(self._patch_dim, self._d_model)
        self._h_pos_embed = nn.Parameter(torch.randn(self._num_patch_h, self._d_model))
        self._w_pos_embed = nn.Parameter(torch.randn(self._num_patch_w, self._d_model))
        self._embed_norm = nn.LayerNorm(self._d_model) if self._embed_layer_norm else None
        self._embed_dropout = nn.Dropout(self._dropout)
        self._layers = nn.ModuleList([TransformerEncoderLayer(d_model=self._d_model,
                                                              nhead=self._n_head,
                                                              dim_feedforward=self._dim_feedforward,
                                                              dropout=self._dropout,
                                                              attention_dropout=self._attention_dropout,
                                                              activation=self._activation,
                                                              normalize_before=self._normalize_before)
                                      for _ in range(self._num_layers)])
        self._norm = nn.LayerNorm(self._d_model) if self._normalize_before else None

    def _forward(self, img: Tensor):
        r"""
        Args:
            img: tokens in src side.
              :math:`(N, H, W, 3)` where N is the batch size, H is the image height, W is the image width and 3 is the rgb channel.

        Outputs:
            - source token hidden representation.
              :math:`(S, N, E)` where S is the source sequence length, N is the batch size,
              E is the embedding size.
        """
        bsz = img.shape[0]
        img = img.reshape(bsz, self._num_patch_h, self._patch_size, self._num_patch_w, self._patch_size, 3)
        img = img.transpose(2, 3).reshape(bsz, self._num_patch_h, self._num_patch_w, self._patch_dim)
        print(img.size())
        x = self._embed(img)
        x = x + self._h_pos_embed[None, :, None, :] + self._w_pos_embed[None, None, :, :]
        x = x.reshape(bsz, self._num_patch, self._d_model)
        x = torch.cat([self._cls_token.repeat((bsz, 1, 1)), x], dim=1)
        if self._embed_norm is not None:
            x = self._embed_norm(x)
        x = self._embed_dropout(x)

        x = x.transpose(0, 1)
        for layer in self._layers:
            x = layer(x)

        if self._norm is not None:
            x = self._norm(x)

        return x[1:], x[0]

    @property
    def d_model(self):
        return self._d_model

    @property
    def out_dim(self):
        return self._d_model
