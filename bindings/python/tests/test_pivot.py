# type: ignore
from slimt.utils import to_json
import json
import pytest


def tokens(response):
    text = response["text"]
    annotations = response["annotation"][0]
    print(text, annotations)
    tokens_ = []
    for annotation in annotations:
        if annotation[0] == annotation[1]:
            continue
        tokens_.append(text[annotation[0] : annotation[1]])
    return tokens_


def jaccard_similarity(source, target):
    source_tokens = tokens(source)
    target_tokens = tokens(target)
    tokens1 = set(source_tokens)
    tokens2 = set(target_tokens)
    common_words = len(tokens1.intersection(tokens2))
    total_words = len(tokens1.union(tokens2))
    return common_words / total_words


def test_pivot(service, models, sample):
    source, _, html = sample

    # Ignore test fails for html sources.
    if html:
        pytest.xfail("html sources are expected to fail")

    responses_ = []
    for model in models:
        responses = service.translate(model, [source], html=html)
        for response in responses:
            responseJSON = json.loads((to_json(response, indent=4)))
            source = responseJSON["target"]["text"]
            responses_.append(responseJSON)
    sources_response = responses_[0]["source"]
    target_response = responses_[-1]["target"]
    assert jaccard_similarity(sources_response, target_response) >= 0.3
