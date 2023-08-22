import os
import subprocess

if __name__ == "__main__":
    browsermt = os.path.join(
        os.getenv("HOME"), ".local/share/bergamot/models/browsermt"
    )

    input_file = "data/numbers2x3.txt"

    models = [
        # fmt: off
        ("enfr.student.tiny11", "model.intgemm.alphas.bin", "vocab.fren.spm", "lex.s2t.bin"),
        ("ende.student.tiny11", "model.intgemm.alphas.bin", "vocab.deen.spm", "lex.s2t.bin"),
        ("enet.student.tiny11", "model.intgemm.alphas.bin", "vocab.eten.spm", "lex.s2t.bin"),
        ("enes.student.tiny11", "model.intgemm.alphas.bin", "vocab.esen.spm", "lex.s2t.bin")
        # fmt: on
    ]

    for bundle in models:
        folder, model, vocab, shortlist = bundle
        model_path = os.path.join(browsermt, folder)
        cmd = [
            # fmt: off
            "./build/bin/slimt",
            "--root", model_path,
            "--model", model,
            "--vocabulary", vocab,
            "--shortlist", shortlist,
            "<", input_file
            # fmt: on
        ]

        print(" ".join(cmd))
