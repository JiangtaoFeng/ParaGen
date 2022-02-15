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
                 labels,
                 path=None):
        super().__init__(path)
        self._encoder_config = encoder
        self._labels = labels

        self._encoder, self._classifier = None, None
        self._path = path

    def _build(self):
        """
        Build model with vocabulary size and special tokens
        """
        self._build_encoder()
        self._build_classifier()

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
        _, x = self.encoder(input)
        logits = self.classifier(x)
        return logits

