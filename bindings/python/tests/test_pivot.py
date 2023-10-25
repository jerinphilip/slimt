# type: ignore
from slimt.utils import toJSON
import json


def jaccard_similarity(string1, string2):
    lowercase_string1 = string1.lower()
    lowercase_string2 = string2.lower()
    words_set1 = set(lowercase_string1.split())
    words_set2 = set(lowercase_string2.split())
    common_words = len(words_set1.intersection(words_set2))
    total_words = len(words_set1.union(words_set2))
    return common_words / total_words


def test_pivot(service_instance, model_instances, source_and_target):
    service = service_instance
    models = model_instances
    source, target = source_and_target
    for model in models:
        responses = service.translate(model, [source], html=False)
        for response in responses:
            source = json.loads(toJSON(response, indent=4))["target"]["text"]
    assert jaccard_similarity(source, target) > 0.3
