
save_dir="/project/efs/users/jwieting/checkpoints/transformer-big"
dir = "/project/efs/users/jwieting/checkpoints"

seed = [1]
obj_alpha = [0.05]
lenpen = [0.5]
smoothing_alpha = [2]

"""
SIM
"""
sim_cmd = "rm -Rf {0} && mkdir {0} && cp {1}/checkpoint_best.pt {0} && NCCL_LL_THRESHOLD=0 " \
          "python -u train.py data-bin/wmt17_en_de -a transformer_vaswani_wmt_en_de_big -s de -t en --max-update 5000 " \
          "--optimizer adam --lr 1e-5 --dropout 0.3 --max-tokens 251 --seq-max-len-a 1.5 --seq-max-len-b 5 " \
          "--criterion sequence_risk --task translation_struct --seq-beam 5 --reset-optimizer " \
          "--save-dir {0} --restore-file checkpoint_best.pt --smoothing-alpha {2} --lr-scheduler polynomial_decay " \
          "--warmup-updates 4000 --weight-decay 0.1 --adam-eps 1e-06 --clip-norm 0.0 --reset-dataloader --reset-meters " \
          "--obj-alpha {3} --distributed-world-size 4 --update-freq 8 --save-interval-updates 2500 --simile-lenpen {4} " \
          "--seq-scorer simile --seq-remove-bpe '@@ ' --seed {5}"
for sa in smoothing_alpha:
      for oa in obj_alpha:
            for lp in lenpen:
                  for s in seed:
                        new_dir = dir + "/simile-{0}-{1}-{2}-{3}".format(sa, oa, lp, s)
                        print(sim_cmd.format(new_dir, save_dir, sa, oa, lp, s))

"""
BLEU
"""
bleu_cmd = "rm -Rf {0} && mkdir {0} && cp {1}/checkpoint_best.pt {0} && NCCL_LL_THRESHOLD=0 " \
           "python -u train.py data-bin/wmt17_en_de -a transformer_vaswani_wmt_en_de_big -s de -t en --max-update 5000 " \
           "--optimizer adam --lr 1e-5 --dropout 0.3 --max-tokens 251 --seq-max-len-a 1.5 --seq-max-len-b 5 " \
           "--criterion sequence_risk --task translation_struct --seq-beam 5 --reset-optimizer " \
           "--save-dir {0} --restore-file checkpoint_best.pt --smoothing-alpha {2} --lr-scheduler polynomial_decay " \
           "--warmup-updates 4000 --weight-decay 0.1 --adam-eps 1e-06 --clip-norm 0.0 --reset-dataloader --reset-meters " \
           "--obj-alpha {3} --distributed-world-size 4 --update-freq 8 --save-interval-updates 2500 " \
           "--seq-remove-bpe '@@ ' --seed {4}"

for sa in smoothing_alpha:
      for oa in obj_alpha:
            for s in seed:
                  new_dir = dir + "/bleu-{0}-{1}-{2}".format(sa, oa, s)
                  print(bleu_cmd.format(new_dir, save_dir, sa, oa, s))

"""
X-SIM
"""
cl_ratio = [0, 0.5, 1]

x_sim_cmd = "rm -Rf {0} && mkdir {0} && cp {1}/checkpoint_best.pt {0} && NCCL_LL_THRESHOLD=0 " \
      "python -u train.py data-bin/wmt17_en_de -a transformer_vaswani_wmt_en_de_big -s de -t en --max-update 5000 " \
      "--optimizer adam --lr 1e-5 --dropout 0.3 --max-tokens 251 --seq-max-len-a 1.5 --seq-max-len-b 5 " \
      "--criterion sequence_risk --task translation_struct --seq-beam 5 --reset-optimizer " \
      "--save-dir {0} --restore-file checkpoint_best.pt --smoothing-alpha {2} --lr-scheduler polynomial_decay " \
      "--warmup-updates 4000 --weight-decay 0.1 --adam-eps 1e-06 --clip-norm 0.0 --reset-dataloader --reset-meters " \
      "--obj-alpha {3} --distributed-world-size 4 --update-freq 8 --save-interval-updates 2500 " \
      "--simile-lenpen {4} --seq-scorer cl-simile --cl-ratio {5} --seq-remove-bpe '@@ ' --seed {6}"

for sa in smoothing_alpha:
      for oa in obj_alpha:
            for lp in lenpen:
                  for cl in cl_ratio:
                        for s in seed:
                              new_dir = dir + "/x-sim-{0}-{1}-{2}-{3}-{4}".format(sa, oa, lp, cl, s)
                              print(x_sim_cmd.format(new_dir, save_dir, sa, oa, lp, cl, s))
