from transformers import OPTConfig, OPTForCausalLM

from paragen.models import register_model
from paragen.models.abstract_model import AbstractModel


@register_model
class HuggingfaceOPTModel(AbstractModel):
    """
    HuggingfacePretrainBartModel is a pretrained bart model built on
    huggingface pretrained bart models.
    """

    def __init__(
        self,
        arch=None,
        pretrained_model=None
    ):
        super().__init__()
        self._arch = arch
        self._pretrained_model = pretrained_model
        if self._pretrained_model is not None:
            if arch is None:
                self._arch = self._pretrained_model
            else:
                assert self._arch == self._pretrained_model
        assert self._arch is not None

        self._config = None
        self._model = None
        self._special_tokens = None

    def _build(self, vocab_size: None, special_tokens: None):
        """
        Build model with vocabulary size and special tokens

        Args:
            vocab_size: vocabulary size of input sequence
            special_tokens: special tokens of input sequence
        """
        assert self._pretrained_model is not None or (vocab_size is not None and special_tokens is not None)
        self._config = OPTConfig.from_pretrained(self._arch)

        if self._pretrained_model is not None:
            self._model = OPTForCausalLM.from_pretrained(self._pretrained_model)
        else:
            self._config.vocab_size = vocab_size
            self._config.pad_token_id, self._config.bos_token_id, self.eos_token_id = special_tokens['pad'], special_tokens['bos'], special_tokens['eos']
            self._model = OPTForCausalLM(self._config)
        self._special_tokens = special_tokens

    def forward(self, src_tokens):
        """
        Compute output with neural input

        Args:
            src_tokens: encoder input sequence

        Returns:
            - log probability of next tokens in sequences
        """
        output = self._model(src_tokens)
        return output.logits
