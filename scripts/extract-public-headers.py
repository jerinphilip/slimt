"""
Usage: 

    grep "slimt/.*.hh" $(ls slimt/*.hh) \
            | python3 scripts/extract-public-headers.py

"""
import os
import sys


def reachable(adj, node):
    includes = set()
    if node in adj:
        for child in adj[node]:
            sub_includes = reachable(adj, child)
            includes = includes.union(sub_includes)
        includes = includes.union(set(adj[node]))
    return includes.union(set([node]))


def partition(headers):
    source = []
    generated = []
    for header in headers:
        if os.path.exists(header):
            source.append(header)
        else:
            generated.append(header)
    return source, generated


if __name__ == "__main__":
    adj = {}
    for line in sys.stdin:
        line = line.strip()
        fname, include = line.split(":")
        include = include.replace('#include "', "")
        include = include.replace('"', "")
        if not fname in adj:
            adj[fname] = []

        adj[fname].append(include)

    public_headers = reachable(adj, "slimt/slimt.hh")
    public_headers = sorted(public_headers)
    source, generated = partition(public_headers)
    print("\n".join(source))
    print("\n")
    print("\n".join(generated))
