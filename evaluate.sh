fairseq-generate data-bin/iwslt14.tokenized.de-en  \
  --path checkpoints/base/checkpoint_best.pt \
  --beam 5 --batch-size 128 --remove-bpe | tee /tmp/gen.out
grep ^H /tmp/gen.out | cut -f3- > /tmp/gen.out.sys
grep ^T /tmp/gen.out | cut -f2- > /tmp/gen.out.ref
fairseq-score --sys /tmp/gen.out.sys --ref /tmp/gen.out.ref

