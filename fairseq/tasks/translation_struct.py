# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import os

import numpy as np
import torch

from fairseq import bleu, utils
from fairseq.data import Dictionary, language_pair_dataset
from fairseq.sequence_generator import SequenceGenerator
from fairseq.tasks import register_task, translation
from .sim_models import WordAveraging
from .sim_utils import Example

from sacremoses import MosesDetokenizer
from nltk.tokenize import TreebankWordTokenizer

class BleuScorer(object):

    key = 'bleu'

    def __init__(self, tgt_dict, bpe_symbol='@@ ', args=None):
        self.tgt_dict = tgt_dict
        self.bpe_symbol = bpe_symbol
        self.scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
        # use a fresh Dictionary for scoring, so that we can add new elements
        self.scoring_dict = Dictionary()

    def preprocess_ref(self, ref):
        ref = self.tgt_dict.string(ref, bpe_symbol=self.bpe_symbol, escape_unk=True)
        return self.scoring_dict.encode_line(ref, add_if_not_exist=True)

    def preprocess_hypo(self, hypo):
        hypo = hypo['tokens']
        hypo = self.tgt_dict.string(hypo.int().cpu(), bpe_symbol=self.bpe_symbol)
        return self.scoring_dict.encode_line(hypo, add_if_not_exist=True)

    def get_cost(self, ref, hypo):
        self.scorer.reset(one_init=True)
        self.scorer.add(ref, hypo)
        return 1. - (self.scorer.score() / 100.)

    def postprocess_costs(self, costs):
        return costs

class SimileScorer(object):

    key = 'simile'

    def __init__(self, tgt_dict, bpe_symbol='@@ ', args=None):
        self.tgt_dict = tgt_dict
        self.bpe_symbol = bpe_symbol
        #self.scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
        model = torch.load('sim/sim.pt',
                               map_location='cpu')

        state_dict = model['state_dict']
        vocab_words = model['vocab_words']
        sim_args = model['args']

        #turn off gpu
        sim_args.gpu = -1

        self.model = WordAveraging(sim_args, vocab_words, sp_file="sim/sim.sp.30k.model")
        self.model.load_state_dict(state_dict, strict=True)
        # use a fresh Dictionary for scoring, so that we can add new elements
        self.scoring_dict = Dictionary()
        self.detok = MosesDetokenizer('en')
        self.tok = TreebankWordTokenizer()

    def preprocess_ref(self, ref):
        ref = self.tgt_dict.string(ref, bpe_symbol=self.bpe_symbol, escape_unk=True)
        return ref

    def preprocess_hypo(self, hypo):
        hypo = hypo['tokens']
        hypo = self.tgt_dict.string(hypo.int().cpu(), bpe_symbol=self.bpe_symbol)
        return hypo

    def get_cost(self, ref, hypo):

        def make_example(sentence):
            sentence = self.detok.detokenize(sentence.split())
            sentence = sentence.lower()
            sentence = " ".join(self.tok.tokenize(sentence))
            sentence = self.model.sp.EncodeAsPieces(sentence)
            wp1 = Example(" ".join(sentence))
            wp1.populate_embeddings(self.model.vocab)
            return wp1

        ref_e = make_example(ref)
        hyp_e = make_example(hypo)
        wx1, wl1, wm1 = self.model.torchify_batch([ref_e])
        wx2, wl2, wm2 = self.model.torchify_batch([hyp_e])
        scores = self.model.scoring_function(wx1, wm1, wl1, wx2, wm2, wl2)

        return scores[0].item()

    def get_costs(self, ref, hypos):

        def make_example(sentence):
            sentence = self.detok.detokenize(sentence.split())
            sentence = sentence.lower()
            sentence = " ".join(self.tok.tokenize(sentence))
            sentence = self.model.sp.EncodeAsPieces(sentence)
            wp1 = Example(" ".join(sentence))
            wp1.populate_embeddings(self.model.vocab)
            return wp1

        ref_e = make_example(ref)
        hypos = [make_example(i) for i in hypos]
        wx1, wl1, wm1 = self.model.torchify_batch([ref_e])
        wx2, wl2, wm2 = self.model.torchify_batch(hypos)
        scores = self.model.scoring_function(wx1, wm1, wl1, wx2, wm2, wl2)

        return scores

    def postprocess_costs(self, costs):
        return costs

class CrossLingualSimileScorer():

    key = 'cl-simile'

    def __init__(self, tgt_dict, src_dict, cl_ratio, bpe_symbol='@@ ', args=None):
        self.tgt_dict = tgt_dict
        self.src_dict = src_dict
        self.bpe_symbol = bpe_symbol
        self.cl_ratio = cl_ratio

        model = torch.load('cl_sim/model.de.lc.100_4_50000.pt',
                               map_location='cpu')

        state_dict = model['state_dict']
        vocab_words = model['vocab']
        sim_args = model['args']

        #turn off gpu
        sim_args.gpu = -1

        self.model = WordAveraging(sim_args, vocab_words, sp_file="cl_sim/all.de.lc.sp.50k.model")
        self.model.load_state_dict(state_dict, strict=True)
        # use a fresh Dictionary for scoring, so that we can add new elements
        self.scoring_dict = Dictionary()
        self.detok = MosesDetokenizer('en')

        self.lower_case = sim_args.lower_case

    def preprocess_ref(self, ref):
        ref = self.tgt_dict.string(ref, bpe_symbol=self.bpe_symbol, escape_unk=True)
        return ref

    def preprocess_hypo(self, hypo):
        hypo = hypo['tokens']
        hypo = self.tgt_dict.string(hypo.int().cpu(), bpe_symbol=self.bpe_symbol)
        return hypo

    def preprocess_src(self, src):
        src = self.src_dict.string(src, bpe_symbol=self.bpe_symbol, escape_unk=True)
        return src

    def get_cost(self, ref, hypo, src):

        def make_example(sentence):
            sentence = self.detok.detokenize(sentence.split())
            if self.lower_case:
                sentence = sentence.lower()
            sentence = self.model.sp.EncodeAsPieces(sentence)
            wp1 = Example(" ".join(sentence), lower=self.lower_case)
            wp1.populate_embeddings(self.model.vocab)
            return wp1

        if self.cl_ratio == 0:
            ref_e = make_example(ref)
            hyp_e = make_example(hypo)
            wx1, wl1, wm1 = self.model.torchify_batch([ref_e])
            wx2, wl2, wm2 = self.model.torchify_batch([hyp_e])
            score = self.model.scoring_function(wx1, wm1, wl1, wx2, wm2, wl2)
            score = score.item()
        elif self.cl_ratio == 1:
            src_e = make_example(src)
            hyp_e = make_example(hypo)
            import pdb
            pdb.set_trace()
            wx1, wl1, wm1 = self.model.torchify_batch([src_e])
            wx2, wl2, wm2 = self.model.torchify_batch([hyp_e])
            score = self.model.scoring_function(wx1, wm1, wl1, wx2, wm2, wl2)
            score = score.item()
        else:
            ref_e = make_example(ref)
            src_e = make_example(src)
            hyp_e = make_example(hypo)
            wx1, wl1, wm1 = self.model.torchify_batch([ref_e])
            wx2, wl2, wm2 = self.model.torchify_batch([hyp_e])
            score_ref = self.model.scoring_function(wx1, wm1, wl1, wx2, wm2, wl2)
            score_ref = score_ref.item()

            wx1, wl1, wm1 = self.model.torchify_batch([src_e])
            score_src = self.model.scoring_function(wx1, wm1, wl1, wx2, wm2, wl2)
            score_src = score_src.item()

            score = self.cl_ratio * score_src + (1 - self.cl_ratio) * score_ref

        return score

    def get_costs(self, ref, hypos, src):

        def make_example(sentence):
            sentence = self.detok.detokenize(sentence.split())
            if self.lower_case:
                sentence = sentence.lower()
            sentence = self.model.sp.EncodeAsPieces(sentence)
            wp1 = Example(" ".join(sentence), lower=self.lower_case)
            wp1.populate_embeddings(self.model.vocab)
            return wp1

        if self.ratio == 0:
            ref_e = make_example(ref)
            hypos = [make_example(i) for i in hypos]
            wx1, wl1, wm1 = self.model.torchify_batch([ref_e])
            wx2, wl2, wm2 = self.model.torchify_batch(hypos)
            scores = self.model.scoring_function(wx1, wm1, wl1, wx2, wm2, wl2)
        elif self.ratio == 1:
            src_e = make_example(src)
            hypos = [make_example(i) for i in hypos]
            wx1, wl1, wm1 = self.model.torchify_batch([src_e])
            wx2, wl2, wm2 = self.model.torchify_batch(hypos)
            scores = self.model.scoring_function(wx1, wm1, wl1, wx2, wm2, wl2)
        else:
            ref_e = make_example(ref)
            src_e = make_example(src)
            hypos = [make_example(i) for i in hypos]
            wx1, wl1, wm1 = self.model.torchify_batch([ref_e])
            wx2, wl2, wm2 = self.model.torchify_batch(hypos)
            score_ref = self.model.scoring_function(wx1, wm1, wl1, wx2, wm2, wl2)

            wx1, wl1, wm1 = self.model.torchify_batch([src_e])
            score_src = self.model.scoring_function(wx1, wm1, wl1, wx2, wm2, wl2)

            scores = self.cl_ratio * score_src + (1 - self.cl_ratio) * score_ref

        return scores

    def postprocess_costs(self, costs):
        return costs


@register_task('translation_struct')
class TranslationStructuredPredictionTask(translation.TranslationTask):
    """
    Translate from one (source) language to another (target) language.

    Compared to :class:`TranslationTask`, this version performs
    generation during training and computes sequence-level losses.

    Args:
        src_dict (Dictionary): dictionary for the source language
        tgt_dict (Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`train.py <train>`,
        :mod:`generate.py <generate>` and :mod:`interactive.py <interactive>`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        translation.TranslationTask.add_args(parser)
        parser.add_argument('--seq-beam', default=5, type=int, metavar='N',
                            help='beam size for sequence training')
        parser.add_argument('--seq-keep-reference', default=False, action='store_true',
                            help='retain the reference in the list of hypos')
        parser.add_argument('--seq-scorer', default='bleu', metavar='SCORER',
                            choices=['bleu', 'simile', 'mixed', 'cl-simile'],
                            help='optimization metric for sequence level training')

        parser.add_argument('--seq-gen-with-dropout', default=False, action='store_true',
                            help='use dropout to generate hypos')
        parser.add_argument('--seq-max-len-a', default=0, type=float, metavar='N',
                            help='generate sequences of maximum length ax + b, '
                                 'where x is the source length')
        parser.add_argument('--seq-max-len-b', default=200, type=int, metavar='N',
                            help='generate sequences of maximum length ax + b, '
                                 'where x is the source length')
        parser.add_argument('--seq-remove-bpe', nargs='?', const='@@ ', default=None,
                            help='remove BPE tokens before scoring')
        parser.add_argument('--seq-sampling', default=False, action='store_true',
                            help='use sampling instead of beam search')
        parser.add_argument('--seq-unkpen', default=0, type=float,
                            help='unknown word penalty to be used in seq generation')
        parser.add_argument('--simile-lenpen', default=0.25, type=float,
                            help='unknown word penalty to be used in seq generation')
        parser.add_argument('--mixed-ratio', default=0.5, type=float,
                            help='unknown word penalty to be used in seq generation')
        parser.add_argument('--cl-ratio', default=0.0, type=float,
                            help='unknown word penalty to be used in seq generation')

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.args = args
        self._generator = None
        self._scorers = {}

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        return super(TranslationStructuredPredictionTask, cls).setup_task(args, **kwargs)

    def build_criterion(self, args):
        """
        Build the :class:`~fairseq.criterions.FairseqCriterion` instance for
        this task.

        Args:
            args (argparse.Namespace): parsed command-line arguments

        Returns:
            a :class:`~fairseq.criterions.FairseqCriterion` instance
        """
        from fairseq import criterions
        criterion = criterions.build_criterion(args, self)
        assert isinstance(criterion, criterions.FairseqSequenceCriterion)
        return criterion

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        # control dropout during generation
        model.train(self.args.seq_gen_with_dropout)

        # generate hypotheses
        self._generate_hypotheses(model, sample)

        return super().train_step(
            sample=sample,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            ignore_grad=ignore_grad,
        )

    def valid_step(self, sample, model, criterion):
        model.eval()
        self._generate_hypotheses(model, sample)
        return super().valid_step(sample=sample, model=model, criterion=criterion)

    def _generate_hypotheses(self, model, sample):
        # initialize generator
        if self._generator is None:
            self._generator = SequenceGenerator(
                self.target_dictionary,
                beam_size=self.args.seq_beam,
                max_len_a=self.args.seq_max_len_a,
                max_len_b=self.args.seq_max_len_b,
                unk_penalty=self.args.seq_unkpen,
                sampling=self.args.seq_sampling,
            )

        # generate hypotheses
        sample['hypos'] = self._generator.generate(
            [model],
            sample,
        )

        # add reference to the set of hypotheses
        if self.args.seq_keep_reference:
            self.add_reference_to_hypotheses(sample)

    def add_reference_to_hypotheses_(self, sample):
        """
        Add the reference translation to the set of hypotheses. This can be
        called from the criterion's forward.
        """
        if 'includes_reference' in sample:
            return
        sample['includes_reference'] = True
        target = sample['target']
        pad_idx = self.target_dictionary.pad()
        for i, hypos_i in enumerate(sample['hypos']):
            # insert reference as first hypothesis
            ref = utils.strip_pad(target[i, :], pad_idx)
            hypos_i.insert(0, {
                'tokens': ref,
                'score': None,
            })

    def get_new_sample_for_hypotheses(self, orig_sample):
        """
        Extract hypotheses from *orig_sample* and return a new collated sample.
        """
        ids = orig_sample['id'].tolist()
        pad_idx = self.source_dictionary.pad()
        samples = [
            {
                'id': ids[i],
                'source': utils.strip_pad(orig_sample['net_input']['src_tokens'][i, :], pad_idx),
                'target': hypo['tokens'],
            }
            for i, hypos_i in enumerate(orig_sample['hypos'])
            for hypo in hypos_i
        ]
        return language_pair_dataset.collate(
            samples, pad_idx=pad_idx, eos_idx=self.source_dictionary.eos(),
            left_pad_source=self.args.left_pad_source, left_pad_target=self.args.left_pad_target,
            sort=False,
        )

    def get_sequence_scorer(self, scorer):
        if scorer not in self._scorers:
            tgt_dict = self.target_dictionary
            src_dict = self.source_dictionary
            if scorer == 'bleu':
                self._scorers[scorer] = BleuScorer(
                    tgt_dict, bpe_symbol=self.args.seq_remove_bpe,
                )
            elif scorer == 'simile':
                self._scorers[scorer] = SimileScorer(
                    tgt_dict, bpe_symbol=self.args.seq_remove_bpe, args=self.args,
                )
            elif scorer == 'cl-simile':
                self._scorers[scorer] = CrossLingualSimileScorer(
                    tgt_dict, src_dict, self.args.cl_ratio, bpe_symbol=self.args.seq_remove_bpe, args=self.args,
                )
            else:
                raise ValueError('Unknown sequence scorer {}'.format(scorer))
        return self._scorers[scorer]

    def get_costs(self, sample, scorer=None):
        """Get costs for hypotheses using the specified *scorer*."""
        if scorer is None:
            scorer = self.get_sequence_scorer(self.args.seq_scorer)

        bsz = len(sample['hypos'])
        nhypos = len(sample['hypos'][0])
        target = sample['target'].int()
        source = sample['net_input']['src_tokens'].int()

        pad_idx = self.target_dictionary.pad()

        costs = torch.zeros(bsz, nhypos).to(sample['target'].device)

        if self.args.seq_scorer == "simile":
            for i, hypos_i in enumerate(sample['hypos']):
                ref = utils.strip_pad(target[i, :], pad_idx).cpu()
                ref = scorer.preprocess_ref(ref)
                ref_len = len(ref.split())
                hypos = []
                hypo_lens = []

                for j, hypo in enumerate(hypos_i):
                    hyp = scorer.preprocess_hypo(hypo)
                    hypos.append(hyp)
                    hypo_lens.append(len(hyp.split()))

                _costs = scorer.get_costs(ref, hypos)

                for j, _ in enumerate(hypos_i):
                    lp = np.exp(1 - max(ref_len, hypo_lens[j]) / float(min(ref_len, hypo_lens[j])))
                    costs[i, j] = 1 - lp ** self.args.simile_lenpen * _costs[j].item()
        elif self.args.seq_scorer == "cl-simile":
            for i, hypos_i in enumerate(sample['hypos']):
                ref = utils.strip_pad(target[i, :], pad_idx).cpu()
                ref = scorer.preprocess_ref(ref)
                src = utils.strip_pad(source[i, :], pad_idx).cpu()
                src = scorer.preprocess_src(src)

                ref_len = len(ref.split())
                hypos = []
                hypo_lens = []

                for j, hypo in enumerate(hypos_i):
                    hyp = scorer.preprocess_hypo(hypo)
                    hypos.append(hyp)
                    hypo_lens.append(len(hyp.split()))

                _costs = scorer.get_costs(ref, hypos, src)

                for j, _ in enumerate(hypos_i):
                    lp = np.exp(1 - max(ref_len, hypo_lens[j]) / float(min(ref_len, hypo_lens[j])))
                    costs[i, j] = 1 - lp ** self.args.simile_lenpen * _costs[j].item()
        else:
            for i, hypos_i in enumerate(sample['hypos']):
                ref = utils.strip_pad(target[i, :], pad_idx).cpu()
                ref = scorer.preprocess_ref(ref)
                for j, hypo in enumerate(hypos_i):
                    costs[i, j] = scorer.get_cost(ref, scorer.preprocess_hypo(hypo))
        return scorer.postprocess_costs(costs)
