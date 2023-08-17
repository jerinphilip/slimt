from argparse import ArgumentParser


def sort_lines(sources_shuf):
    with open(sources_shuf) as sources_shuf_file:
        lines = sources_shuf_file.read().splitlines()
        sorted_lines = sorted(lines, key=lambda x: len(x))
        return sorted_lines


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("sources_shuf", type=str, help="Path to sources.shuf")
    args = parser.parse_args()

    output_fpath = args.sources_shuf + ".sorted"
    lines = sort_lines(args.sources_shuf)
    with open(output_fpath, "w+") as output_file:
        output_file.write("\n".join(lines))
