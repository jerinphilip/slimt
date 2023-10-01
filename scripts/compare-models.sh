#!bin/bash

MODELS=(
  "enfr.student.tiny11"
  "ende.student.tiny11"
  "enet.student.tiny11"
)

for model in ${MODELS[@]}; do
  python3 scripts/marian-file-inspect.py \
    --model-path $HOME/.local/share/bergamot/models/browsermt/$model/model.intgemm.alphas.bin \
    > traces/$model.model.txt
done
