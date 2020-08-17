base_cmd = "python -u train.py data-bin/wmt17_en_de -a transformer_vaswani_wmt_en_de_big --optimizer adam " \
           "--lr 0.0005 -s de -t en --label-smoothing 0.1 --max-tokens 3125 --update-freq 2 --min-lr '1e-09' " \
           "--lr-scheduler inverse_sqrt --weight-decay 0.0001 --criterion label_smoothed_cross_entropy " \
           "--distributed-world-size 4 --warmup-updates 4000 --warmup-init-lr '1e-07' --adam-betas '(0.9, 0.98)' " \
           "--save-dir checkpoints/transformer-big --max-epoch 50"
print(base_cmd)
