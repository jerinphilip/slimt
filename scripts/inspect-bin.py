import os
import sys
from argparse import ArgumentParser
from ast import literal_eval

import numpy as np


def Tensor(fpath, shape=None, dtype=None):
    if not shape:
        return None
        fname = os.path.basename(fpath)
        prefix, ext = args.file.split(".")
        fields = prefix.split("-")

    if not dtype:
        print(fields)
        return None
        op, type_val, shape, shape_val, size, size_val, which_side = prefix.split("_")
        return None
        sizes = list(map(int, shape_val.split("x")))

    mapping = {"float32": np.float32, "uint32": np.uint32}

    A = np.fromfile(args.file, dtype=mapping[dtype], count=-1, sep="")
    A = np.reshape(A, shape)
    return A


if __name__ == "__main__":
    parser = ArgumentParser()
    choices = ["i32", "i8", "f32"]
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--type", type=str)
    parser.add_argument("--shape", type=str)
    args = parser.parse_args()

    if args.shape is not None:
        args.shape = literal_eval(args.shape)

    print(Tensor(args.file, args.shape, args.type))
