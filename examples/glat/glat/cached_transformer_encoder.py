from torch import Tensor

from paragen.modules.encoders import register_encoder
from paragen.modules.encoders.transformer_encoder import TransformerEncoder


@register_encoder
class CachedTransformerEncoder(TransformerEncoder):
    """
    TransformerEncoder is a transformer encoder.

    Args:
        num_layers: number of encoder layers
        d_model: feature dimension
        n_head: head numbers of multihead attention
        dim_feedforward: dimensionality of inner vector space
        dropout: dropout rate
        activation: activation function used in feed-forward network
        learn_pos: learning postional embedding instead of sinusoidal one
        return_seed: return with sequence representation
        normalize_before: use pre-norm fashion, default as post-norm.
            Pre-norm suit deep nets while post-norm achieve better results when nets are shallow.
        name: module name
    """

    def __init__(self,
                 *args,
                 **kwargs):
        super(CachedTransformerEncoder, self).__init__(*args, **kwargs)
        self.cache = {}

    def _forward(self, src: Tensor):
        r"""
        Args:
            src: tokens in src side.
              :math:`(N, S)` where N is the batch size, S is the source sequence length.

        Outputs:
            - source token hidden representation.
              :math:`(S, N, E)` where S is the source sequence length, N is the batch size,
              E is the embedding size.
        """
        x = self._embed(src)
        self.cache['token_embed'] = x
        if self._embed_scale is not None:
            x = x * self._embed_scale
        if self._pos_embed is not None:
            x = x + self._pos_embed(src)
        if self._embed_norm is not None:
            x = self._embed_norm(x)
        x = self._embed_dropout(x)

        src_padding_mask = src.eq(self._special_tokens['pad'])
        x = x.transpose(0, 1)
        for layer in self._layers:
            x = layer(x, src_key_padding_mask=src_padding_mask)

        if self._norm is not None:
            x = self._norm(x)

        if self._return_seed:
            encoder_out = x[1:], src_padding_mask[:, 1:], x[0]
        else:
            encoder_out = x, src_padding_mask

        return encoder_out

    def get(self, name):
        return self.cache[name]

    def reset(self, mode):
        super(CachedTransformerEncoder, self).reset(mode)
        self.cache.clear()
