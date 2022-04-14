from paragen.models import register_model

from examples.glat.glat.glat import GLATModel


@register_model
class DSLPModel(GLATModel):

    @property
    def dslp_logits(self):
        return self._decoder.dslp_logits
