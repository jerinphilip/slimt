import yaml
import argparse
import textwrap
import itertools


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
        blob_path = f'blob_path("{self.save}")'
        return f'tensor_from_file<{dtype}>({blob_path}, {shape}, "{self.name}")'


def NoOp(lhs, rhs):
    return ""


def test(lhs, rhs, slimt_fn):
    block = []
    args = ", ".join([arg.name for arg in rhs])
    info = f"{lhs.name} == {slimt_fn}({args})"
    block.append(f'std::string info = "{info}";')
    block.append(f"Tensor lhs_expected = {lhs.load()};")
    for idx, arg in enumerate(rhs):
        block.append(f"Tensor rhs_{idx} = {rhs[idx].load()};")
    args = ", ".join([f"rhs_{idx}" for idx in range(len(rhs))])
    block.append(f"Tensor lhs_computed = {slimt_fn}({args});")
    block.append(f'CHECK_EQUAL(lhs_computed, lhs_expected, "{info}");')
    return "{\n" + "\n".join(block) + "\n}"


def guard(block):
    catch_block = """catch (const std::exception& e) {
        // Catching and handling exceptions
        std::cerr << "Exception caught: " << e.what() << std::endl;
    }
    catch (...) {{
        // Catching any other unexpected exceptions
        std::cerr << "Unknown exception caught" << std::endl;
    }}
    """
    return f"""try {{ {block} }} {catch_block}"""


def ReLU(lhs, rhs):
    lhs.reshape([prod(lhs.shape)])
    for arg in rhs:
        arg.reshape([prod(arg.shape)])
    return test(lhs, rhs, "relu")


def Plus(lhs, rhs):
    lhs.reshape([prod(lhs.shape)])
    for arg in rhs:
        if prod(arg.shape) != prod(lhs.shape):
            return ""
        arg.reshape([prod(arg.shape)])
    return test(lhs, rhs, "add")


def Highway(lhs, rhs):
    lhs.reshape([prod(lhs.shape)])
    blocks = []
    for arg in rhs:
        if prod(arg.shape) != prod(lhs.shape):
            return ""
        arg.reshape([prod(arg.shape)])
    block = test(lhs, rhs, "highway")
    blocks.append(block)
    return "\n".join(blocks)


def LayerNormalization(lhs, rhs):
    return test(lhs, rhs, "layer_norm")


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
    blocks = list(map(guard, blocks))
    return textwrap.dedent(
        """
        #include "TestSuite.hh"
        using namespace slimt; // NOLINT
    int main(){{
    {}
    return 0;
    }}""".format(
            "\n\n".join(blocks)
        )
    )


# Mappings from marian to slimt
mapping = {
    # "AffineNodeOp": Affine,
    "ColsNodeOp": NoOp,
    "ConstantNode": NoOp,
    "cpu:integer:AffineNodeOp<marian:Type:int8>": NoOp,
    "cpu:integer:DotNodeOp<marian:Type:int8>": NoOp,
    "cpu:integer:PrepareANodeOp<marian:Type:int8>": NoOp,
    "cpu:integer:QuantMultNodeOp<marian:Type:int8>": NoOp,
    "DotBatchedNodeOp": NoOp,
    "GatherNodeOp": NoOp,
    "HighwayNodeOp": Highway,
    "LayerNormalizationOp": LayerNormalization,
    "LogSoftmaxNodeOp": NoOp,
    "NegNodeOp": NoOp,
    "ParamNode": NoOp,
    "PlusNodeOp": Plus,
    "ReLUNodeOp": ReLU,
    "RowsNodeOp": NoOp,
    "ScalarAddNodeOp": NoOp,
    "ScalarMultNodeOp": NoOp,
    "SoftmaxNodeOp": NoOp,
    "TransposeNodeOp": NoOp,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    data = None
    with open(args.trace) as fp:
        data = yaml.safe_load(fp)

    blocks = Blocks(mapping, data)
    with open(args.output, "w") as output:
        print(main(blocks), file=output)
