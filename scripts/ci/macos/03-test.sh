# Install `bergamot` CLI via pip.
python3 -m pip install bergamot -f https://github.com/jerinphilip/bergamot-translator/releases/expanded_assets/latest

# Download en-de-tiny and de-en-tiny models.
bergamot download -m en-de-tiny

BROWSERMT="$HOME/Library/Application Support/bergamot/models/browsermt/"
PREFIX="$BROWSERMT/ende.student.tiny11"

MODEL=model.intgemm.alphas.bin
VOCAB=vocab.deen.spm
SHORTLIST=lex.s2t.bin

# ./build/app/slimt-cli --root "${PREFIX}" \
#   --model ${MODEL} --vocabulary ${VOCAB} --shortlist ${SHORTLIST} \
#   < data/sample.txt

For cursed debugging on CI
lldb --batch \
  -o "settings set target.input-path ${PWD}/data/sample.txt" \
  -o "settings set -- target.run-args --root \"${PREFIX}\" --model \"${MODEL}\" --vocabulary \"${VOCAB}\" --shortlist \"${SHORTLIST}\"" \
  -o "process launch" \
  -o "bt" -o "exit" \
  -- ./build/app/slimt-cli
