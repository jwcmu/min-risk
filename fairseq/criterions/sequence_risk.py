# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch.nn.functional as F

from fairseq import utils
from fairseq.criterions import FairseqSequenceCriterion, register_criterion

@register_criterion('sequence_risk')
class SequenceRiskCriterion(FairseqSequenceCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

        from fairseq.tasks.translation_struct import TranslationStructuredPredictionTask
        if not isinstance(task, TranslationStructuredPredictionTask):
            raise Exception(
                'sequence_risk criterion requires `--task=translation_struct`'
            )
        self.smoothing_alpha = args.smoothing_alpha
        self.rescale_costs = args.rescale_costs
        self.eps = 0.1
        self.obj_alpha = args.obj_alpha

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--normalize-costs', action='store_true',
                            help='normalize costs within each hypothesis')
        parser.add_argument('--smoothing-alpha', type=float, default=1.0,
                            help='normalize costs within each hypothesis')
        parser.add_argument('--no-rescale-costs', action='store_false',
                            help='normalize costs within each hypothesis')
        parser.add_argument('--obj-alpha', type=float, default=0.0,
                            help='normalize costs within each hypothesis')
        # fmt: on

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        non_pad_mask = target.ne(self.padding_idx)
        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        bsz = len(sample['hypos'])
        nhypos = len(sample['hypos'][0])

        # get costs for hypotheses using --seq-scorer (defaults to 1. - BLEU)
        costs = self.task.get_costs(sample)
        if self.rescale_costs:
            costs *= 100

        if self.args.normalize_costs:
            unnormalized_costs = costs.clone()
            max_costs = costs.max(dim=1, keepdim=True)[0]
            min_costs = costs.min(dim=1, keepdim=True)[0]
            costs = (costs - min_costs) / (max_costs - min_costs).clamp_(min=1e-6)
        else:
            unnormalized_costs = None

        # generate a new sample from the given hypotheses
        new_sample = self.task.get_new_sample_for_hypotheses(sample)
        hypotheses = new_sample['target'].view(bsz, nhypos, -1, 1)
        hypolen = hypotheses.size(2)
        pad_mask = hypotheses.ne(self.task.target_dictionary.pad())
        lengths = pad_mask.sum(dim=2).float()

        mle = model(**sample['net_input'])
        mle_loss, _ = self.compute_loss(model, mle, sample, reduce=reduce)
        net_output = model(**new_sample['net_input'])
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(bsz, nhypos, hypolen, -1)

        scores = lprobs.gather(3, hypotheses)
        scores *= pad_mask.float()
        avg_scores = scores.sum(dim=2) / lengths
        probs = F.softmax(self.smoothing_alpha*avg_scores, dim=1).squeeze(-1)
        risk_loss = (probs * costs).sum()

        loss = (1 - self.obj_alpha) * risk_loss + self.obj_alpha * mle_loss
        sample_size = bsz
        assert bsz == utils.item(costs.size(dim=0))
        logging_output = {
            'loss': utils.item(loss.data),
            'risk_loss': utils.item(risk_loss),
            'mle_loss': utils.item(mle_loss),
            'num_cost': costs.numel(),
            'ntokens': sample['ntokens'],
            'nsentences': bsz,
            'sample_size': sample_size,
        }

        def add_cost_stats(costs, prefix=''):
            logging_output.update({
                prefix + 'sum_cost': utils.item(costs.sum()),
                prefix + 'min_cost': utils.item(costs.min(dim=1)[0].sum()),
                prefix + 'cost_at_1': utils.item(costs[:, 0].sum()),
            })

        add_cost_stats(costs)
        if unnormalized_costs is not None:
            add_cost_stats(unnormalized_costs, 'unnormalized_')

        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        num_costs = sum(log.get('num_cost', 0) for log in logging_outputs)
        agg_outputs = {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size,
            'mle': sum(log.get('mle_loss', 0) for log in logging_outputs) / sample_size,
            'risk': sum(log.get('risk_loss', 0) for log in logging_outputs) / sample_size,
            'ntokens': ntokens,
            'nsentences': nsentences,
        }

        def add_cost_stats(prefix=''):
            agg_outputs.update({
                prefix + 'avg_cost': sum(log.get(prefix + 'sum_cost', 0) for log in logging_outputs) / num_costs,
                prefix + 'min_cost': sum(log.get(prefix + 'min_cost', 0) for log in logging_outputs) / nsentences,
                prefix + 'cost_at_1': sum(log.get(prefix + 'cost_at_1', 0) for log in logging_outputs) / nsentences,
            })

        add_cost_stats()
        if any('unnormalized_sum_cost' in log for log in logging_outputs):
            add_cost_stats('unnormalized_')

        return agg_outputs
