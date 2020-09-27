import os
import torch
import sentencepiece as spm
import argparse
import numpy as np
from fairseq.tasks.sim_utils import Example
from fairseq.tasks.sim_models import WordAveraging
from sacremoses import MosesDetokenizer
from nltk.tokenize import TreebankWordTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--sim-model-file', default="simile-mrt/sim/sim.pt",
                help='Model file for SIM.')
parser.add_argument('--sim-sp-file', default="simile-mrt/sim/sim.sp.30k.model",
                help='SP file for SIM.')
parser.add_argument('--length-penalty', type=float, default=0.25, metavar='D',
                help='Weight of length penalty on SIM term.')
parser.add_argument('--sys-file', help='path to save checkpoints')
parser.add_argument('--ref-file', help='path to save checkpoints')

args = parser.parse_args()

def make_example(sentence, detok, tok, sp):
    sentence = detok.detokenize(sentence.split())
    sentence = sentence.lower()
    sentence = " ".join(tok.tokenize(sentence))
    sentence = sp.EncodeAsPieces(sentence)
    return " ".join(sentence)

def score_output(args):
    sp = spm.SentencePieceProcessor()
    sp.Load(args.sim_sp_file)

    detok = MosesDetokenizer('en')
    tok = TreebankWordTokenizer()

    f_sys = open(args.sys_file,'r')
    lines_sys = f_sys.readlines()

    f_ref = open(args.ref_file,'r')
    lines_ref = f_ref.readlines()
    
    bleu_pairs = []
    sim_pairs = []

    for i in zip(lines_sys, lines_ref):
        s = i[0].strip()
        r = i[1].strip()
        s_sim = make_example(s, detok, tok, sp)
        r_sim = make_example(r, detok, tok, sp)
        bleu_pairs.append((s, r))
        sim_pairs.append((s_sim, r_sim))

    model = torch.load(args.sim_model_file,
                           map_location='cpu')

    state_dict = model['state_dict']
    vocab_words = model['vocab_words']
    sim_args = model['args']
    model = WordAveraging(sim_args, vocab_words)
    model.load_state_dict(state_dict, strict=True)

    scores = []
    scores_simile = []
    for idx, i in enumerate(sim_pairs):
        wp1 = Example(i[0])
        wp1.populate_embeddings(model.vocab)
        wp2 = Example(i[1])
        wp2.populate_embeddings(model.vocab)
        wx1, wl1, wm1 = model.torchify_batch([wp1])
        wx2, wl2, wm2 = model.torchify_batch([wp2])
        score = model.scoring_function(wx1, wm1, wl1, wx2, wm2, wl2)
        ref_l = len(bleu_pairs[idx][0].split())
        hyp_l = len(bleu_pairs[idx][1].split())
        lp = np.exp(1 - max(ref_l, hyp_l) / float(min(ref_l, hyp_l)))
        simile = lp ** args.length_penalty * score.item()
        scores_simile.append(simile)
        scores.append(score.item())

    print("SIM: {0}".format(np.mean(scores)))
    print("SimiLe: {0}".format(np.mean(scores_simile)))

score_output(args)
