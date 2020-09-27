import torch
import argparse
import numpy as np

import sys
sys.path.insert(0, "/home/jwieting/min-risk-cl-simile/min-risk")

from fairseq.tasks.sim_utils import Example
from fairseq.tasks.sim_models import WordAveraging
from sacremoses import MosesDetokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--length-penalty', type=float, default=0.25, metavar='D',
                    help='Weight of length penalty on SIM term.')
parser.add_argument('--sys-file', help='path to save checkpoints')
parser.add_argument('--src-file', help='path to save checkpoints')

args = parser.parse_args()

def score_output(args):
    detok = MosesDetokenizer('en')

    model = torch.load('simile-mrt/cl_sim/wmt.all.lc.sp.50k.model',
                       map_location='cpu')

    state_dict = model['state_dict']
    vocab_words = model['vocab']
    sim_args = model['args']

    # turn off gpu
    sim_args.gpu = -1

    model = WordAveraging(sim_args, vocab_words, sp_file="simile-mrt/cl_sim/model.wmt.all.lc.100.0.0_25.pt")
    model.load_state_dict(state_dict, strict=True)
    # use a fresh Dictionary for scoring, so that we can add new elements
    lower_case = sim_args.lower_case

    def make_example(sentence):
        sentence = detok.detokenize(sentence.split())
        if lower_case:
            sentence = sentence.lower()
        sentence = model.sp.EncodeAsPieces(sentence)
        wp1 = Example(" ".join(sentence), lower=lower_case)
        wp1.populate_embeddings(model.vocab)
        return wp1

    f_sys = open(args.sys_file, 'r')
    lines_sys = f_sys.readlines()

    f_src = open(args.src_file, 'r')
    lines_src = f_src.readlines()
    
    sim_pairs = []

    for i in zip(lines_sys, lines_src):
        s = i[0].strip()
        r = i[1].strip()
        s_sim = make_example(s)
        r_sim = make_example(r)
        sim_pairs.append((s_sim, r_sim))

    scores = []
    for idx, i in enumerate(sim_pairs):
        wp1 = i[0]
        wp2 = i[1]
        wx1, wl1, wm1 = model.torchify_batch([wp1])
        wx2, wl2, wm2 = model.torchify_batch([wp2])
        score = model.scoring_function(wx1, wm1, wl1, wx2, wm2, wl2)
        scores.append(score.item())

    print("XLSIM-SIM: {0}".format(np.mean(scores)))

score_output(args)
