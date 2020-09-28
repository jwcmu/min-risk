import os
from glob import glob
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=["all", "list", "evaluate"], default="all")
parser.add_argument('--file', default=None)
parser.add_argument('--name', default=None)

args = parser.parse_args()

def evaluate(file, name):
    
    os.system("fairseq-generate simile-mrt/data-bin/wmt17_en_de --path {0} -s de -t en --beam 5 --batch-size 128 --remove-bpe --gen-subset valid > {1}.gen.valid.out".format(file, name))
    os.system("grep ^H {0}.gen.valid.out | cut -f3- > {0}.gen.valid.out.sys".format(name))
    os.system("grep ^T {0}.gen.valid.out | cut -f2- > {0}.gen.valid.out.ref".format(name))
    os.system("grep ^S {0}.gen.valid.out | cut -f2- > {0}.gen.valid.out.src".format(name))
    os.system("fairseq-score --sys {0}.gen.valid.out.sys --ref {0}.gen.valid.out.ref --sacrebleu > {0}.txt".format(name))
    os.system("python evaluate_bleu_sim.py --sys-file {0}.gen.valid.out.sys --ref-file {0}.gen.valid.out.ref >> {0}.txt".format(name))
    os.system("python evaluate_xlsim.py --sys-file {0}.gen.valid.out.sys --src-file {0}.gen.valid.out.src >> {0}.txt".format(name))

    os.system("echo \"\" >> {0}.txt".format(name))
    
    os.system("fairseq-generate simile-mrt/data-bin/wmt17_en_de --path {0} -s de -t en --beam 5 --batch-size 128 --remove-bpe > {1}.gen.test.out".format(file, name))
    os.system("grep ^H {0}.gen.test.out | cut -f3- > {0}.gen.test.out.sys".format(name))
    os.system("grep ^T {0}.gen.test.out | cut -f2- > {0}.gen.test.out.ref".format(name))
    os.system("grep ^S {0}.gen.test.out | cut -f2- > {0}.gen.test.out.src".format(name))
    os.system("fairseq-score --sys {0}.gen.test.out.sys --ref {0}.gen.test.out.ref --sacrebleu >> {0}.txt".format(name))
    os.system("python evaluate_bleu_sim.py --sys-file {0}.gen.test.out.sys --ref-file {0}.gen.test.out.ref >> {0}.txt".format(name))
    os.system("python evaluate_xlsim.py --sys-file {0}.gen.test.out.sys --src-file {0}.gen.test.out.src >> {0}.txt".format(name))

    """
    dirlist = ["out-of-domain/indomain_training/data-bin/", "out-of-domain/QED/data-bin/",
               "out-of-domain/open-subtitles/data-bin/", "out-of-domain/ted/data-bin/"]

    names = ["it", "qed", "os", "ted"]
    
    for i,j in zip(dirlist, names):
        os.system(
            "fairseq-generate {0} --path {1} -s de -t en --beam 5 --batch-size 128 --remove-bpe | tee ood.gen.out".format(
                i, file))
        os.system("grep ^H ood.gen.out | cut -f3- > ood.gen.out.sys".format(name))
        os.system("grep ^T ood.gen.out | cut -f2- > ood.gen.out.ref".format(name))
        os.system("grep ^S ood.gen.out | cut -f2- > ood.gen.out.src".format(name))
        os.system("echo {0} >> {1}.txt".format(j, name))
        os.system("fairseq-score --sys ood.gen.out.sys --ref ood.gen.out.ref --sacrebleu >> {0}.txt".format(name))
        os.system("python evaluate_bleu_sim.py --sys-file ood.gen.out.sys --ref-file ood.gen.out.ref >> {0}.txt".format(
            name))
        os.system(
            "python evaluate_xlsim.py --sys-file ood.gen.out.sys --src-file ood.gen.out.src >> {0}.txt".format(name))
    """

dirs = glob("checkpoints/*")

if args.mode == "all":
    for d in dirs:
        name = d.replace("checkpoints/","")

        file = d + "/checkpoint_1_5000.pt"
        if os.path.exists(file):
            evaluate(file, name + "-{0}".format(5000))

        file = d + "/checkpoint_1_10000.pt"
        if os.path.exists(file):
            evaluate(file, name + "-{0}".format(10000))
elif args.mode == "list":
    for d in dirs:
        name = d.replace("checkpoints/","")

        file = d + "/checkpoint_1_5000.pt"
        if os.path.exists(file):
            print("python evaluate_ablations.py --mode evaluate --file {0} --name {1}".format(file, name + "-{0}".format(5000)))

        file = d + "/checkpoint_1_10000.pt"
        if os.path.exists(file):
            print("python evaluate_ablations.py --mode evaluate --file {0} --name {1}".format(file, name + "-{0}".format(10000)))
elif args.mode == "evaluate":
    if os.path.exists(args.file):
        evaluate(args.file, args.name + "-{0}".format(5000))

    if os.path.exists(args.file):
        evaluate(args.file, args.name + "-{0}".format(10000))