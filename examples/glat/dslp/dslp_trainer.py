import torch
try:
    import horovod.torch as hvd
except:
    pass
from paragen.trainers.trainer import Trainer
from paragen.trainers import register_trainer
from paragen.criteria import create_criterion

from examples.glat.glat.glat_trainer import GLATTrainer


@register_trainer
class DSLPTrainer(GLATTrainer):

    def __init__(self, dslp_criterion, *args, **kwargs):
        super(DSLPTrainer, self).__init__(*args, **kwargs)
        self._dslp_criterion_config = dslp_criterion

        self._dslp_criterion = None

    def build(self, *args, **kwargs):
        super(DSLPTrainer, self).build(*args, **kwargs)
        self._dslp_criterion = create_criterion(self._dslp_criterion_config)
        self._dslp_criterion.build(self._model, self._tgt_special_tokens['pad'])

    def _forward_loss(self, samples):
        """
        Train one batch of samples

        Args:
            samples: a batch of samples
        Returns:
            logging_states: states to display in progress bar
        """
        self._model.reset(mode='train')
        self._model.set_seed(self._tot_step_cnt)
        glancing_output = self._generator(**samples['net_input'])

        glancing_target = samples.pop('glancing_target')
        masked_target, fusing_target_mask = self.glancing(
            glancing_output,
            **glancing_target
        )
        samples['net_input']['target'] = glancing_target['target']
        samples['net_input']['fusing_target_mask'] = fusing_target_mask
        samples['net_output']['token']['target'] = masked_target

        loss, logging_states = self._criterion(**samples)

        layer, bsz, seqlen, vocab_size = self._model.dslp_logits.shape
        dslp_logits = self._model.dslp_logits.reshape(layer * bsz, seqlen, vocab_size)
        dslp_target = samples['net_output']['token']['target'][None].repeat(layer, 1, 1).reshape(layer * bsz, seqlen)
        dslp_loss, dslp_logging_states = self._dslp_criterion.compute_loss(dslp_logits, dslp_target)
        dslp_logging_states = {f'dslp.{k}': v for k, v in dslp_logging_states.items()}
        logging_states.update(dslp_logging_states)

        loss = loss + dslp_loss * layer

        logging_states['total_loss'] = loss.data.item()

        if torch.isnan(loss).any():
            logging_states = {}
        return loss, logging_states
