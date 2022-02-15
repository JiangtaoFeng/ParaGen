from typing import Dict

from paragen.criteria import create_criterion
from paragen.models import create_model
from paragen.tasks import register_task
from paragen.tasks.base_task import BaseTask
from paragen.utils.data import reorganize
from paragen.utils.tensor import create_tensor, convert_tensor_to_idx


@register_task
class ViTTask(BaseTask):

    def __init__(self,
                 **kwargs):
        super(ViTTask, self).__init__(**kwargs)

    def _build_models(self):
        """
        Build a text classification model
        """
        self._model = create_model(self._model_configs)
        self._model.build()

    def _build_criterions(self):
        """
        Build a criterion
        """
        self._criterion = create_criterion(self._criterion_configs)
        self._criterion.build(self._model)

    def _collate(self, samples: Dict, is_training=False) -> Dict:
        samples = reorganize(samples)
        images, labels = samples['image'], samples['label']
        images_t = create_tensor(images, float)
        labels_t = create_tensor(labels, int)
        batch = {
            'net_input': {
                'input': images_t
            },
            'net_output': {
                'target': labels_t
            }
        }
        if self._infering:
            batch['text_output'] = labels
        return batch

    def _output_collate_fn(self, outputs, *args, **kwargs):
        outputs = convert_tensor_to_idx(outputs)
        processed_outputs = []
        for output in outputs:
            processed_outputs.append(output)
        return processed_outputs
