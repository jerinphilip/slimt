import struct
from argparse import ArgumentParser
import difflib


class Bijective:
    def __init__(self, forward):
        self.forward = forward
        self.backward = {value: key for key, value in self.forward.items()}

    def get(self, key):
        return self.forward.get(key, None)

    def inverse(self, key):
        return self.backward.get(key, None)


# fmt: off
TYPE_CLASS = Bijective({
    "signed_type"   : int("0x0100", 16),
    "unsigned_type" : int("0x0200", 16),
    "float_type"    : int("0x0400", 16),
    "packed_type"   : int("0x0800", 16),  # special packed (CPU cache friendly) type class, used in FBGEMM. Annoyingly we need to keep 0x800 for back-compat, would be nicer to align with intgemm
    "avx2_type"     : int("0x1000", 16),  # processor-specific layout for avx2, currently used for FBGEMM only (keep 0x1000 for back-compat)
    "avx512_type"   : int("0x2000", 16),  # processor-specific layout for avx512, currently used for FBGEMM only (keep 0x2000 for back-compat)
    "intgemm_type"  : int("0x4000", 16),  # intgemm quantized architecture agnostic models
    "size_mask"     : int("0x00FF", 16),  # maximum allowed size is 256 bytes right now; if more are required, extend the size field
    "class_mask"    : int("0xFF00", 16),  # three fields for different type classes, if more classes are added we need to increase the number of fields here
})

TYPE = Bijective({
    "int8"          : TYPE_CLASS.get("signed_type")   + 1,                                  # int8 type
    "int16"         : TYPE_CLASS.get("signed_type")   + 2,                                  # int16 type
    "int32"         : TYPE_CLASS.get("signed_type")   + 4,                                  # int32 type
    "int64"         : TYPE_CLASS.get("signed_type")   + 8,                                  # int64 type

    "uint8"         : TYPE_CLASS.get("unsigned_type") + 1,                                  # uint8 type
    "uint16"        : TYPE_CLASS.get("unsigned_type") + 2,                                  # uint16 type
    "uint32"        : TYPE_CLASS.get("unsigned_type") + 4,                                  # uint33 type
    "uint64"        : TYPE_CLASS.get("unsigned_type") + 8,                                  # uint64 type

    "float16"       : TYPE_CLASS.get("float_type")    + 2,                                  # float16 type
    "float32"       : TYPE_CLASS.get("float_type")    + 4,                                  # float32 type
    "float64"       : TYPE_CLASS.get("float_type")    + 8,                                  # float64 type

    "packed16"      : TYPE_CLASS.get("packed_type")   + 2,                                  # special type for FBGEMM, not meant to be used anywhere else, not meant to be accessed invidually. Internal actual type (uint16) is meaningless.
    "packed8avx2"   : TYPE_CLASS.get("packed_type")   + 2 + TYPE_CLASS.get("avx2_type"),    # special type for FBGEMM with AVX2, not meant to be used anywhere else, not meant to be accessed invidually. Internal actual type (uint8) is meaningless.
    "packed8avx512" : TYPE_CLASS.get("packed_type")   + 1 + TYPE_CLASS.get("avx512_type"),  # special type for FBGEMM with AVX512, not meant to be used anywhere else, not meant to be accessed invidually. Internal actual type (uint8) is meaningless.

    "intgemm8"      : TYPE_CLASS.get("signed_type")   + 1 + TYPE_CLASS.get("intgemm_type"), # Int8 quantized (not packed) matrices for intgemm
    "intgemm16"     : TYPE_CLASS.get("signed_type")   + 2 + TYPE_CLASS.get("intgemm_type"), # Int16 quantized (not packed) matrices for intgemm
})
# fmt: on


class Header:
    def __init__(self, name_len, type_id, shape_len, data_len):
        self.name_len = name_len
        self.shape_len = shape_len
        self.type = type_id
        self.data_len = data_len

    def __repr__(self):
        return f"Header(name={self.name_len}, type={self.type}, shape={self.shape_len}, data={self.data_len})"


class Reader:
    def __init__(self, fp):
        self.fp = fp

    def uint64(self):
        buffer = self.fp.read(8)
        arg, *_ = struct.unpack("<Q", buffer)
        return arg

    def int32(self):
        buffer = self.fp.read(4)
        arg, *_ = struct.unpack("<i", buffer)
        return arg

    def char(self):
        buffer = self.fp.read(1)
        byte, *_ = struct.unpack("<c", buffer)
        arg = byte.decode("utf-8")
        return arg

    def string(self, length):
        binary = self.fp.read(length)
        return binary.decode("utf-8")[:-1]

    def bytes(self, length):
        data = self.fp.read(length)
        return data

    def header(self):
        num_fields = 4
        args = [self.uint64() for i in range(num_fields)]
        return Header(*args)

    def shape(self, length):
        data = []
        for i in range(length):
            datum = self.int32()
            data.append(datum)
        return data


class Item:
    def __init__(self, type_id):
        self.name = None
        self.shape = None
        self.data = None
        self.type = type_id

    def set_shape(self, shape):
        self.shape = shape

    def set_data(self, data):
        self.data = data

    def set_name(self, name):
        self.name = name

    def __repr__(self):
        data_len = len(self.data)
        typename = TYPE.inverse(self.type) or f"UnknownType[{self.type}]"
        return f"Item({self.name}, {typename}, Shape({self.shape}), {data_len})"


def parse(model_path):
    with open(model_path, "rb") as fp:
        reader = Reader(fp)
        version = reader.uint64()
        num_headers = reader.uint64()

        headers = []
        for i in range(num_headers):
            header = reader.header()
            headers.append(header)

        items = [Item(header.type) for header in headers]
        for header, item in zip(headers, items):
            name = reader.string(header.name_len)
            item.set_name(name)

        for header, item in zip(headers, items):
            shape = reader.shape(header.shape_len)
            item.set_shape(shape)

        # To align by 256, some slack is left. This distance is indicated in the
        # next byte. Read, then consume distance bytes.
        distance_to_256_aligned = reader.uint64()

        garbage = reader.bytes(distance_to_256_aligned)

        for header, item in zip(headers, items):
            data = reader.bytes(header.data_len)
            item.set_data(data)

        for header, item in zip(headers, items):
            config_name = "special:model.yml"
            print(item)
            if item.name == config_name:
                config = item.data.decode("utf-8")[:-1]
                print(config)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    args = parser.parse_args()
    parse(args.model_path)
