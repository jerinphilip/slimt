import os
import subprocess as sp
import sys
from argparse import ArgumentParser
from collections.abc import Iterable, Mapping
from copy import deepcopy

import yaml


def str_presenter(dumper, data):
    """configures yaml for dumping multiline strings
    Ref: https://stackoverflow.com/questions/8640959/how-can-i-control-what-scalar-form-pyyaml-uses-for-my-data
    """
    if data.count("\n") > 0:  # check for multiline string
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


# to use with safe_dump:
yaml.add_representer(str, str_presenter)
yaml.representer.SafeRepresenter.add_representer(str, str_presenter)


def recurse_substitute(node, lhs, rhs):
    if isinstance(node, str):
        return node.replace(lhs, rhs)
    if isinstance(node, Mapping):
        tx = {}
        for key, value in node.items():
            tx[key] = recurse_substitute(value, lhs, rhs)
        return tx
    if isinstance(node, Iterable):
        tx = []
        for child in node:
            tx.append(recurse_substitute(child, lhs, rhs))
        return tx

    return node


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--github-workflow", type=str, required=True)
    parser.add_argument("--tx-out", type=str, default=None)
    args = parser.parse_args()
    with open(args.github_workflow) as workflow_file:
        config = yaml.safe_load(workflow_file)
        config = recurse_substitute(config, "${{ github.workspace }}", "/tmp/slimt")
        env = deepcopy(config["env"])
        p = sp.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True)
        env["PYTHON_LOCAL_VERSION_IDENTIFIER"] = p.stdout.decode("utf-8")

        def var(name):
            return "${{ " + name + " }}"

        for key, value in env.items():
            config = recurse_substitute(config, var("env.{}".format(key)), str(value))

        # print(config["jobs"]["build-wheels"])

        if args.tx_out:
            with open(args.tx_out, "w+") as transformed_file:
                yaml.dump(config, transformed_file)

        steps = config["jobs"]["build-wheels"]["steps"]
        for step in steps:
            if step.get("name", None) == "Build wheels":
                env = step["env"]
                for key, value in env.items():
                    os.environ[key] = str(value)

        cmd_args = ["python3", "-m", "cibuildwheel", "--platform", "linux"]
        sp.run(cmd_args)
