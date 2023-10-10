import subprocess as sp
import sys
from subprocess import PIPE, STDOUT, Popen


def run_successfully(cmd, stdin=None):
    print(" ".join(cmd))
    if stdin is not None:
        p = Popen(cmd, stdout=PIPE, stdin=PIPE, stderr=PIPE)
        stdin = stdin.encode()
        stdout_data = p.communicate(input=stdin)[0]
        p.stdin.close()
        p.wait()
        return stdout_data
    else:
        p = sp.run(cmd, capture_output=True)
        p.check_returncode()
        return p.stdout


def test_cmdline():
    python = sys.executable
    base = [python, "-m", "slimt"]

    for model in ["en-de-tiny", "de-en-tiny"]:
        run_successfully(base + ["download", "-m", model])

    run_successfully(base + ["ls"])

    opus_args = ["-r", "opus"]
    run_successfully(base + ["download", "-m", "eng-fin-tiny"] + opus_args)
    run_successfully(base + ["ls"] + opus_args)
    # Run the sample python script shipped with module

    for model in ["en-de-tiny", "de-en-tiny"]:
        data = run_successfully(
            base + ["translate", "--model", model], stdin="Hello World"
        )
        print(data)

    data = run_successfully(
        base + ["translate", "--model", "eng-fin-tiny"] + opus_args,
        stdin="Hello World",
    )
    print(data)
