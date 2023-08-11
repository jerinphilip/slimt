import sys
from collections import deque


def transform(line):
    # Easier parsing reverse?
    # Parse file:loc
    i = len(line)

    assert line[i - 1] == ")"

    end = i
    i = i - 1
    while line[i] != "(":
        i = i - 1

    # line[i] = '(', line[i:end] = (...)
    left, right = i + 1, end - 1
    source_location = line[left:right]
    file_name, line_number = source_location.split(":")

    file_url = file_name.replace(
        "/home/jerin/code/bergamot-translator/3rd_party/marian-dev/",
        "https://github.com/jerinphilip/marian/blob/8c4170fa08c46df1cf4c987e493b7a3772c380b3/",
    )

    stripped_source_location = source_location.replace("/home/jerin/code/", "")

    end = i

    # Skip spaces.
    i = i - 1
    while line[i] != ")":
        i = i - 1

    end = i + 1
    stack = deque()
    stack.append(")")
    i = i - 1

    while len(stack) > 0 and i > 0:
        if line[i] == ")":
            stack.append(")")

        if line[i] == "(":
            stack.pop()

        i = i - 1

    begin = i + 1
    args = line[begin:end]
    # print("args=", args)

    left, right = 0, begin
    identifier = line[left:right]

    hyperlinked_identifier = f'<a href="{file_url}#L{line_number}">{identifier}</a>'

    indicate = lambda x: f"[{x}]"
    return f"{hyperlinked_identifier}{args} "


if __name__ == "__main__":
    for line in sys.stdin:
        line = line.strip()
        render = transform(line)
        print(render)
