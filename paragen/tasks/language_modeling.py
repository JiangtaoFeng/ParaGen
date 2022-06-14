from typing import Dict, List

from paragen.criteria import create_criterion
from paragen.datasets import create_dataset
from paragen.generators import create_generator
from paragen.models import create_model
from paragen.tasks import register_task
from paragen.tasks.base_task import BaseTask
from paragen.utils.tensor import convert_idx_to_tensor, convert_tensor_to_idx


@register_task
class LanguageModelingTask(BaseTask):
    """
    Seq2SeqTask defines overall scope on sequence to sequence task.

    Args:
        src: source key in data dict
        tgt: target key in data dict
        lang: task language
        maxlen: maximum length for sequence
        share_vocab: share source and target vocabulary
        index_only: only do indexing
    """

    def __init__(self,
                 index_only=False,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self._index_only = index_only

    def _build_datasets(self):
        """
        Build a datasets
        """
        self._datasets = {}
        for key, configs in self._data_configs.items():
            dataset = create_dataset(configs)
            if key == 'train':
                dataset.build(collate_fn=lambda x: self._data_collate_fn(x, is_training=True),
                              preprocessed=self._preprocessed,
                              sep_token=self._tokenizer.eos_token)
            else:
                dataset.build(collate_fn=lambda x: self._data_collate_fn(x, is_training=False),
                              preprocessed=self._preprocessed,
                              sep_token=self._tokenizer.eos_token)
            self._datasets[key] = dataset

    def _build_models(self):
        """
        Build a sequence-to-sequence model
        """
        self._model = create_model(self._model_configs)
        self._model.build(vocab_size=len(self._tokenizer),
                          special_tokens=self._tokenizer.special_tokens)

    def _build_criterions(self):
        """
        Build a criterion
        """
        self._criterion = create_criterion(self._criterion_configs)
        self._criterion.build(self._model, padding_idx=self._tokenizer.pad)

    def _build_generator(self):
        """
        Build generator for model in inference
        """
        self._generator = create_generator(self._generator_configs)
        self._generator.build(self._model,
                              special_tokens=self._tokenizer.special_tokens)

    def _data_collate_fn(self, sample: str, is_training=True) -> Dict:
        sample = sample.strip()
        return {
            'text': sample,
            'token_num': sample.count(' '),
            'processed': self._tokenizer.encode(sample)
        }

    def _collate(self, samples: List[Dict]):
        """
        Create batch from a set of processed samples

        Args:
            a list of samples:

        Returns:
            batch: a processed batch of samples used as neural network inputs
        """
        samples = [sample['processed'] for sample in samples]
        tgt, prev_tokens = [sample[1:] for sample in samples], [sample[:-1] for sample in samples]
        tgt = convert_idx_to_tensor(tgt, pad=self._tokenizer.pad)
        prev_tokens = convert_idx_to_tensor(prev_tokens, pad=self._tokenizer.pad)
        batch = {
            'net_input': {
                'src_tokens': prev_tokens,
            },
            'net_output': {
                'target': tgt
            }
        }
        return batch

    def _output_collate_fn(self, sample, *args, **kwargs):
        """
        Parse decoded results by convert tensor to list

        Returns:
            idx (list): debatched idx
        """
        sample = convert_tensor_to_idx(sample,
                                       bos=self._tokenizer.bos,
                                       eos=self._tokenizer.eos,
                                       pad=self._tokenizer.pad)
        sample = [self._tokenizer.decode(s) for s in sample]
        return sample
