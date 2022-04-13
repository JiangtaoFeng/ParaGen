from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from paragen.modules.decoders import AbstractDecoder, register_decoder
from paragen.modules.decoders.nonauto_transformer_decoder import NonAutoTransformerDecoder
from paragen.modules.layers.bert_layer_norm import BertLayerNorm
from paragen.modules.layers.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding
from paragen.modules.layers.learned_positional_embedding import LearnedPositionalEmbedding
from paragen.modules.decoders.layers.nonauto_transformer_decoder_layer import NonAutoTransformerDecoderLayer


@register_decoder
class DSLPTransformerDecoder(NonAutoTransformerDecoder):

    def __init__(self, *args, **kwargs):
        super(DSLPTransformerDecoder, self).__init__(*args, **kwargs)
        self.dslp_logits = []
        self._projs = None

    def build(self, *args, **kwargs):
        super(DSLPTransformerDecoder, self).build(*args, **kwargs)
        self._projs = nn.ModuleList([
            nn.Linear(self._d_model * 2, self._d_model)
        for _ in range(self._num_layers - 1)])

    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_padding_mask,
                memory_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = tgt
        if self._pos_embed is not None:
            pos_embed = self._pos_embed(tgt_padding_mask.long())
            x = x + pos_embed
        if self._embed_norm is not None:
            x = self._embed_norm(x)
        x = self._embed_dropout(x)

        self.dslp_logits = []
        x = x.transpose(0, 1)
        for i, layer in enumerate(self._layers):
            x = layer(tgt=x,
                      memory=memory,
                      tgt_key_padding_mask=tgt_padding_mask,
                      memory_key_padding_mask=memory_padding_mask, )
            if i < self._num_layers - 1:
                x = self._fusing_hypo(x, self._projs[i])
        x = x.transpose(0, 1)
        self.dslp_logits = torch.cat(self.dslp_logits, dim=0).transpose(1, 2)

        logits = self._out_proj(x)
        return logits

    def _fusing_hypo(self, x, proj):
        logits = self._out_proj(x)
        self.dslp_logits.append(logits[None])
        logits = logits.detach()
        hypo = logits.argmax(dim=-1)
        hypo_embed = self._embed(hypo)
        if self._embed_norm is not None:
            x = self._embed_norm(hypo_embed)
        x = torch.cat([x, hypo_embed], dim=-1)
        x = proj(x)
        return x

