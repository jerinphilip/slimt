import json
import os
import typing as t

import urllib.request
import yaml

from . import Alignments, AnnotatedText, Response, Package
from .typing_utils import URL, PathLike


def download_resource(url: URL, save_location: PathLike, force_download=False):
    """
    Downloads a resource from url into save_location, overwrites only if
    force_download is true.
    """
    if force_download or not os.path.exists(save_location):
        urllib.request.urlretrieve(url, filename=save_location)


def patch_marian_for_slimt(
    marian_config_path: PathLike, slimt_config_path: PathLike, quality: bool = False
):
    """
    Accepts path to a config-file from marian-training and followign
    quantization and adjusts parameters for use in slimt.
    """
    # Load marian_config_path
    data = None
    with open(marian_config_path) as fp:
        data = yaml.load(fp, Loader=yaml.FullLoader)

    # Update a few entries. Things here are hardcode.
    data.update(
        {
            "ssplit-prefix-file": "",
            "ssplit-mode": "paragraph",
            "max-length-break": 128,
            "mini-batch-words": 1024,
            "workspace": 128,  # shipped models use big workspaces. We'd prefer to keep it low.
            "alignment": "soft",
        }
    )

    if quality:
        data.update({"quality": quality, "skip-cost": False})

    # Write-out.
    with open(slimt_config_path, "w") as output_file:
        print(yaml.dump(data, sort_keys=False), file=output_file)


def to_json(response: Response, *args, **kwargs) -> str:
    def to_py_native(annotated_text: AnnotatedText) -> t.Dict[t.Any, t.Any]:
        result = []
        for sentenceIdx in range(annotated_text.sentence_count()):
            wordLs = []
            for wordIdx in range(annotated_text.word_count(sentenceIdx)):
                word = annotated_text.word_as_range(sentenceIdx, wordIdx)
                wordLs.append((word.begin, word.end))
            result.append(wordLs)

        return {"text": annotated_text.text, "annotation": result}

    return json.dumps(
        {
            "source": to_py_native(response.source),
            "target": to_py_native(response.target),
            "alignments": list(response.alignments),
        },
        *args,
        **kwargs
    )


def package_from_config_path(path):
    with open(path) as yaml_file:
        c = yaml.safe_load(yaml_file)
    root = os.path.dirname(path)
    package = Package(
        model=os.path.join(root, c["models"][0]),
        vocabulary=os.path.join(root, c["vocabs"][0]),
        shortlist=os.path.join(root, c["shortlist"][0]),
    )
    return package
