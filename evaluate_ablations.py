import os
from glob import glob

def evaluate(file, name):
    
    os.system("fairseq-generate simile-mrt/data-bin/wmt17_en_de --path {0} -s de -t en --beam 5 --batch-size 128 --remove-bpe --gen-subset valid | tee {1}.gen.out".format(file, name))
    os.system("grep ^H {0}.gen.out | cut -f3- > {0}.gen.out.sys".format(name))
    os.system("grep ^T {0}.gen.out | cut -f2- > {0}.gen.out.ref".format(name))
    os.system("grep ^S {0}.gen.out | cut -f2- > {0}.gen.out.src".format(name))
    os.system("fairseq-score --sys {0}.gen.out.sys --ref {0}.gen.out.ref --sacrebleu > {0}.txt".format(name))
    os.system("python evaluate_bleu_sim.py --sys-file {0}.gen.out.sys --ref-file {0}.gen.out.ref >> {0}.txt".format(name))
    os.system("python evaluate_xlsim.py --sys-file {0}.gen.out.sys --src-file {0}.gen.out.src >> {0}.txt".format(name))
    
    
    os.system("fairseq-generate simile-mrt/data-bin/wmt17_en_de --path {0} -s de -t en --beam 5 --batch-size 128 --remove-bpe | tee {1}.gen.out".format(file, name))
    os.system("grep ^H {0}.gen.out | cut -f3- > {0}.gen.out.sys".format(name))
    os.system("grep ^T {0}.gen.out | cut -f2- > {0}.gen.out.ref".format(name))
    os.system("grep ^S {0}.gen.out | cut -f2- > {0}.gen.out.src".format(name))
    os.system("fairseq-score --sys {0}.gen.out.sys --ref {0}.gen.out.ref --sacrebleu >> {0}.txt".format(name))
    os.system("python evaluate_bleu_sim.py --sys-file {0}.gen.out.sys --ref-file {0}.gen.out.ref >> {0}.txt".format(name))
    os.system("python evaluate_xlsim.py --sys-file {0}.gen.out.sys --src-file {0}.gen.out.src >> {0}.txt".format(name))

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

for d in dirs:
    name = d.replace("checkpoints/","")
    file = d + "/checkpoint_1_5000.pt"
    evaluate(file, name + "-{0}".format(5000))

    file = d + "/checkpoint_1_10000.pt"
    evaluate(file, name + "-{0}".format(10000))
