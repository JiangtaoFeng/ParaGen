from paragen.models import register_model
from paragen.models.abstract_encoder_decoder_model import AbstractEncoderDecoderModel
from paragen.modules.decoders import create_decoder
from paragen.modules.encoders import create_encoder
from paragen.modules.layers.classifier import LinearClassifier
from paragen.modules.utils import create_source_target_modality, uniform_assignment, create_sequence
from paragen.utils.ops import local_seed

from examples.glat.glat.glat import GLATModel


@register_model
class DSLPModel(GLATModel):

    @property
    def dslp_logits(self):
        return self._decoder.dslp_logits
