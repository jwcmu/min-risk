alphas = [2]
mle = [0.05]
cl_ratio = [0, 0.5, 1]
savedir="/project/efs/users/jwieting/checkpoints/transformer-big"
newdir="/project/efs/users/jwieting/checkpoints/transformer-mrt-cl-simile"

base = "python -u train.py data-bin/wmt17_en_de -a transformer_vaswani_wmt_en_de_big " \
       "-s de -t en --max-update 5000 --optimizer adam " \
       "--lr 1e-5 --dropout 0.3 --max-tokens 251 --seq-max-len-a 1.5 " \
       "--seq-max-len-b 5 --criterion sequence_risk --task translation_struct" \
       " --seq-beam 5 --reset-optimizer --save-dir {0} " \
       "--restore-file checkpoint_best.pt --smoothing-alpha {1} --lr-scheduler " \
       "polynomial_decay --warmup-updates 4000 --weight-decay 0.1 --adam-eps 1e-06 " \
       "--clip-norm 0.0 --reset-dataloader --reset-meters --obj-alpha {2} " \
       "--distributed-world-size 4 --update-freq 8 --save-interval-updates 2500 " \
       "--simile-lenpen 0.5 --seq-scorer cl-simile --cl-ratio {3} --seq-remove-bpe '@@ '"

ct=1

for i in alphas:
    for j in mle:
        for k in cl_ratio:
            fname = newdir + "-{0}-{1}-{2}".format(i, j, k)
            cmd = "rm -Rf {0} && mkdir {0} " \
                  "&& cp {1}/checkpoint_best.pt {0} " \
                  "&& NCCL_LL_THRESHOLD=0 ".format(fname, savedir) + base.format(fname, i, j, k)
            print(cmd + " > outfile-cl-simile-{0}.txt 2>&1".format(ct))
            ct += 1
