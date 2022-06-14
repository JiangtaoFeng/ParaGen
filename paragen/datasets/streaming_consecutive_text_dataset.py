from paragen.datasets import register_dataset
from paragen.datasets.streaming_dataset import StreamingDataset
from paragen.utils.data import safe_data
from paragen.utils.io import UniIO


@register_dataset
class StreamingConsecutiveTextDataset(StreamingDataset):
    """
    StreamingTextDataset is a streaming dataset for reading data in textual format.

    Args:
        path: path to load the data
    """

    def __init__(self,
                 path,
                 maxlen,
                 split_mode='sentence'):
        super().__init__(path)
        self._maxlen = maxlen
        self._split_mode = split_mode
        self._sep_token = '<eos>'

    def build(self, collate_fn=None, preprocessed=False, sep_token='<eos>'):
        """
        Build input stream

        Args:
             collate_fn: callback defined by a specific task
        """
        self._collate_fn = collate_fn
        self._sep_token = sep_token

        if self._path:
            self._fin = UniIO(self._path)

    def __iter__(self):
        """
        fetch next sample

        Returns:
            sample: next sample
        """
        tokens = []
        for line in self._fin:
            with safe_data():
                cur_tokens = [self._sep_token] + line.strip('\n').split()
                cur_len = len(cur_tokens)
                if cur_len + len(tokens) >= self._maxlen:
                    if self._split_mode == 'full':
                        tokens += cur_tokens
                        sample = self._full_callback(' '.join(tokens[:self._maxlen]))
                        tokens = tokens[self._maxlen:]
                    else:
                        sample = self._full_callback(' '.join(tokens + [self._sep_token]))
                        tokens = cur_tokens
                    yield sample
                else:
                    tokens += cur_tokens
        if self._split_mode != 'full':
            tokens += [self._sep_token]
        with safe_data():
            yield self._full_callback(' '.join(tokens))

    def reset(self):
        """
        reset the dataset
        """
        self._pos = 0
        self._fin = UniIO(self._path)

    def _callback(self, sample):
        """
        Callback for json data

        Args:
            sample: data in raw format

        Returns:
            sample (dict): a dict of samples consisting of parallel data of different sources
        """
        sample = sample.strip('\n').strip()
        return sample

    def finalize(self):
        """
        Finalize dataset after finish reading
        """
        self._fin.close()

