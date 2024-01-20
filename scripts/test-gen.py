import yaml
import argparse
import textwrap


def prod(xs):
    accumulator = 1
    for x in xs:
        accumulator = accumulator * x
    return accumulator


class Tensor:
    def __init__(self, name, dtype, shape, save):
        self.name = name
        self.dtype = dtype
        self.shape = list(map(int, shape[1:-1].split("x")))
        self.save = save

    def reshape(self, shape):
        assert prod(shape) == prod(self.shape)
        self.shape = shape

    def load(self):
        dims = list(map(str, self.shape))
        shape = "Shape({{{ls}}})".format(ls=", ".join(dims))
        dmap = {"float32": "float", "int8": "int8_t"}
        dtype = dmap[self.dtype]
        name = self.name
        return f'tensor_from_file<{dtype}>("{self.save}", {shape}, "{name}")'


def NoOp(lhs, rhs):
    return ""


def test(lhs, rhs, slimt_fn):
    block = []
    block.append(f"Tensor lhs_expected = {lhs.load()};")
    for idx, arg in enumerate(rhs):
        block.append(f"Tensor rhs_{idx} = {rhs[idx].load()};")
    args = ", ".join([f"rhs_{idx}" for idx in range(len(rhs))])
    block.append(f"Tensor lhs_computed = {slimt_fn}({args});")
    block.append("CHECK_EQUAL(lhs_computed, lhs_expected);")
    return "{\n" + "\n".join(block) + "\n}"


def ReLU(lhs, rhs):
    lhs.reshape([prod(lhs.shape)])
    for arg in rhs:
        arg.reshape([prod(arg.shape)])
    return test(lhs, rhs, "relu")


def Affine(lhs, rhs):
    return test(lhs, rhs, "affine")


def parse(t):
    """Parse a tensor"""
    tensor_id = t["id"]
    save = t["save"]

    name, dtype, shape, *_ = tensor_id.split()
    return Tensor(name, dtype, shape, save)


def emit(op, lhs_info, rhs_info):
    lhs = parse(lhs_info)
    rhs = [parse(arg) for arg in rhs_info]
    return op(lhs, rhs)


def Blocks(mapping, data):
    ls = []
    for entry in data:
        fn = entry["fn"]
        key = ":".join(filter(lambda x: x, fn.split(":")[2:-3]))
        lhs = entry["lhs"]
        rhs = entry.get("rhs", [])
        if key in mapping:
            op = mapping[key]
            codeblock = emit(op, lhs, rhs)
            if codeblock:
                ls.append(codeblock)

    return ls


def main(blocks):
    return textwrap.dedent(
        """
        #include "slimt/TestSuite.hh"
        using namespace slimt; // NOLINT
    int main(){{
    {}
    return 0;
    }}""".format(
            "\n\n".join(blocks)
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=str, required=True)
    args = parser.parse_args()
    data = None
    with open(args.trace) as fp:
        data = yaml.safe_load(fp)

    # Mappings from marian to slimt
    mapping = {
        "AffineNodeOp": Affine,
        "ColsNodeOp": NoOp,
        "ConstantNode": NoOp,
        "cpu:integer:AffineNodeOp<marian:Type:int8>": NoOp,
        "cpu:integer:DotNodeOp<marian:Type:int8>": NoOp,
        "cpu:integer:PrepareANodeOp<marian:Type:int8>": NoOp,
        "cpu:integer:QuantMultNodeOp<marian:Type:int8>": NoOp,
        "DotBatchedNodeOp": NoOp,
        "GatherNodeOp": NoOp,
        "HighwayNodeOp": NoOp,
        "LayerNormalizationOp": NoOp,
        "LogSoftmaxNodeOp": NoOp,
        "NegNodeOp": NoOp,
        "ParamNode": NoOp,
        "PlusNodeOp": NoOp,
        "ReLUNodeOp": ReLU,
        "RowsNodeOp": NoOp,
        "ScalarAddNodeOp": NoOp,
        "ScalarMultNodeOp": NoOp,
        "SoftmaxNodeOp": NoOp,
        "TransposeNodeOp": NoOp,
    }

    blocks = Blocks(mapping, data)
    print(main(blocks))
