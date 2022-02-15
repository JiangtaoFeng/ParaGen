import torch
import torch.nn as nn

from paragen.models import AbstractModel, register_model
from paragen.modules.encoders import create_encoder
from paragen.modules.layers.classifier import LinearClassifier


@register_model
class ViTModel(AbstractModel):
    """
    SequenceClassificationModel is a general sequence classification architecture consisting of
    one encoder and one classifier.

    Args:
        encoder: encoder configuration
        labels: number of labels
        path: path to restore model
    """

    def __init__(self,
                 encoder,
                 d_model,
                 height,
                 width,
                 patch_size,
                 labels,
                 path=None):
        super().__init__(path)
        self._encoder_config = encoder
        self._d_model = d_model
        self._labels = labels
        self._height, self._width = height, width
        self._patch_size = patch_size

        self._embed, self._encoder, self._classifier = None, None, None
        self._path = path

    def _build(self):
        """
        Build model with vocabulary size and special tokens
        """
        self._build_embedding()
        self._build_encoder()
        self._build_classifier()

    def _build_embedding(self):
        self._embed = Embedding2D(self._height, self._width, self._patch_size, self._d_model)

    def _build_encoder(self):
        """
        Build encoder with vocabulary size and special tokens
        """
        self._encoder = create_encoder(self._encoder_config)
        self._encoder.build()

    def _build_classifier(self):
        """
        Build classifer on label space
        """
        self._classifier = LinearClassifier(self.encoder.out_dim, self._labels)

    @property
    def encoder(self):
        return self._encoder

    @property
    def classifier(self):
        return self._classifier

    def forward(self, input):
        """
        Compute output with neural input

        Args:
            input: input image batch

        Returns:
            - log probability of labels
        """
        x = self._embed(input)
        x = self._encoder(x.mean(dim=1))
        logits = self.classifier(x)
        return logits


class Embedding2D(nn.Module):

    def __init__(self, height, width, patch_size, d_model, embed_layer_norm=False, dropout=0.0):
        super().__init__()
        self._height, self._width = height, width
        self._patch_size = patch_size
        self._d_model = d_model

        assert self._height % self._patch_size == 0 and self._width % self._patch_size == 0
        self._num_patch_h, self._num_patch_w = self._height // self._patch_size, self._width // self._patch_size
        self._num_patch = self._num_patch_h * self._num_patch_w
        self._patch_dim = self._patch_size ** 2 * 3

        self._linear_proj = nn.Linear(self._patch_dim, self._d_model)
        self._h_pos_embed = nn.Parameter(torch.randn(self._num_patch_h, self._d_model))
        self._w_pos_embed = nn.Parameter(torch.randn(self._num_patch_w, self._d_model))
        self._embed_norm = nn.LayerNorm(self._d_model) if embed_layer_norm else None
        self._embed_dropout = nn.Dropout(dropout)
        self._cls_token = nn.Parameter(torch.randn(1, 1, self._d_model))

    def forward(self, x):
        bsz = x.shape[0]
        x = x.reshape(bsz, self._num_patch_h, self._patch_size, self._num_patch_w, self._patch_size, 3)
        x = x.transpose(2, 3).reshape(bsz, self._num_patch_h, self._num_patch_w, self._patch_dim)
        x = self._linear_proj(x)
        x = x + self._h_pos_embed[None, :, None, :] + self._w_pos_embed[None, None, :, :]
        x = x.reshape(bsz, self._num_patch, self._d_model)
        x = torch.cat([self._cls_token.repeat((bsz, 1, 1)), x], dim=1)
        if self._embed_norm is not None:
            x = self._embed_norm(x)
        x = self._embed_dropout(x)
        return x
