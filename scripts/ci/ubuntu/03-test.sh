#/bin/bash

# Install `bergamot` CLI via pip.
python3 -m pip install bergamot -f https://github.com/jerinphilip/bergamot-translator/releases/expanded_assets/latest

# Download en-de-tiny and de-en-tiny models.
bergamot download -m en-de-tiny

BROWSERMT=$HOME/.local/share/bergamot/models/browsermt
PREFIX=$BROWSERMT/ende.student.tiny11

MODEL=model.intgemm.alphas.bin
VOCAB=vocab.deen.spm
SHORTLIST=lex.s2t.bin

./build/bin/slimt --root ${PREFIX} \
  --model ${MODEL} --vocabulary ${VOCAB} --shortlist ${SHORTLIST} \
  < data/sample.txt
