checpoint=$1

fairseq-generate simile-mrt/data-bin/wmt17_en_de  \
  --path $checkpoint \
  --beam 5 --batch-size 128 --remove-bpe -s de -t en --gen-subset valid | tee /tmp/gen.out

grep ^H /tmp/gen.out | cut -f3- > /tmp/gen.out.sys
grep ^T /tmp/gen.out | cut -f2- > /tmp/gen.out.ref
fairseq-score --sys /tmp/gen.out.sys --ref /tmp/gen.out.ref --sacrebleu
mv /tmp/gen.out.sys gen.out.sim.sys
mv /tmp/gen.out.ref gen.out.sim.ref