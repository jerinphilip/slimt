#/bin/bash

SLIMT_PSEUDO_WHEEL=1 python3 setup.py install --user
python3 -m slimt download -m en-de-tiny

BROWSERMT=$HOME/.local/share/slimt/models/browsermt
PREFIX=$BROWSERMT/ende.student.tiny11

MODEL=model.intgemm.alphas.bin
VOCAB=vocab.deen.spm
SHORTLIST=lex.s2t.bin

set -x

./build/app/slimt-cli --root ${PREFIX} \
  --model ${MODEL} --vocabulary ${VOCAB} --shortlist ${SHORTLIST} \
  < data/sample.txt
