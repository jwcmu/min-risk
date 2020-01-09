import sys
import os

#dir = sys.argv[1]

def evaluate(dir):
    os.system("fairseq-generate data-bin/iwslt14.tokenized.de-en --path checkpoints/{0}/checkpoint_best.pt --beam 5 --batch-size 128 --remove-bpe | tee checkpoints/{0}/gen.out".format(dir))
    os.system("grep ^H checkpoints/{0}/gen.out | cut -f3- > checkpoints/{0}/gen.out.sys".format(dir))
    os.system("grep ^T checkpoints/{0}/gen.out | cut -f2- > checkpoints/{0}/gen.out.ref".format(dir))
    os.system("fairseq-score --sys checkpoints/{0}/gen.out.sys --ref checkpoints/{0}/gen.out.ref > {0}.txt".format(dir))

from glob import glob

dirs = glob('checkpoints/*')
for i in dirs:
    i = i.replace("checkpoints/","")
    evaluate(i)
