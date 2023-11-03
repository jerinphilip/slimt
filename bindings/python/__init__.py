import sys
import typing
import warnings

try:
    from ._slimt import *  # type: ignore
except ImportError:
    warnings.warn("Error importing libslimt.so, some functions may not work")

    # Stub symbols
    Alignments = None
    Response = None
    Model = None
    Service = None
    AnnotatedText = None
    Package = None
    Config = None


from .repository import Aggregator, TranslateLocallyLike

REPOSITORY = Aggregator(
    [
        TranslateLocallyLike("browsermt", "https://translatelocally.com/models.json"),
        TranslateLocallyLike(
            "opus", "https://object.pouta.csc.fi/OPUS-MT-models/app/models.json"
        ),
    ]
)
"""
REPOSITORY is a global object that aggregates multiple model-providers to
provide a (model-provider: str, model-code: str) based query mechanism to
get models.
"""
