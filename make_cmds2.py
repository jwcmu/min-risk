alphas = [1,0.5,0.1,0.05,0.01]
rescale = [0]

base = "fairseq-train data-bin/iwslt14.tokenized.de-en -a fconv_iwslt_de_en --clip-norm 0.1 --momentum 0.9 --lr 0.25 --dropout 0.3 --max-tokens 500 --seq-max-len-a 1.5 --seq-max-len-b 5 --criterion sequence_risk --task translation_struct --seq-beam 16 --reset-optimizer --save-dir checkpoints/{0} --min-lr 1e-5 --restore-file checkpoint_best.pt --smoothing-alpha {1} --max-epoch 15 --lr-scheduler reduce_lr_on_plateau --lr-shrink 0.5 --reset-dataloader --reset-meters"

for i in alphas:
    for j in rescale:
        fname = "min-risk-{0}-{1}".format(i,j)
        cmd = "rm -Rf checkpoints/{0} && mkdir checkpoints/{0} && cp checkpoints/base/checkpoint_best.pt checkpoints/{0} && ".format(fname) + base.format(fname,i)
        if j:
            cmd += " --rescale-costs"
        print(cmd)
