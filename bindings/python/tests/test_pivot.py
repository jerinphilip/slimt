# type: ignore
import pytest
import yaml
import os
from slimt import REPOSITORY, Model, Service, Package, Config
from slimt.utils import toJSON
import json


def jaccard_similarity(string1, string2):
    lowercase_string1 = string1.lower()
    lowercase_string2 = string2.lower()

    words_set1 = set(lowercase_string1.split())
    words_set2 = set(lowercase_string2.split())

    common_words = len(words_set1 & words_set2)
    total_words = len(words_set1 | words_set2)

    if total_words == 0:
        # Handle the case where both strings are empty
        return 0.0
    else:
        return common_words / total_words


def test_basic():
    keys = ["browsermt"]
    models = ["en-es-tiny", "es-en-tiny"]
    service = Service(workers=1, cache_size=1024)
    for repository in keys:
        for model in models:
            REPOSITORY.download(repository, model)

        source = "How embarrassing. A fridge full of condiments and no food."
        target = str(source)
        for modelId in models:
            config_path = REPOSITORY.modelConfigPath(repository, modelId)
            c = None
            with open(config_path) as yaml_file:
                c = yaml.safe_load(yaml_file)
            print(c, flush=True)
            root = os.path.dirname(config_path)
            package = Package(
                model=os.path.join(root, c["models"][0]),
                vocabulary=os.path.join(root, c["vocabs"][0]),
                shortlist=os.path.join(root, c["shortlist"][0]),
            )
            config = Config()
            model = Model(config, package)
            print(repository, modelId)
            responses = service.translate(model, [source], html=False)
            for response in responses:
                source = json.loads(toJSON(response, indent=4))["target"]["text"]
        assert jaccard_similarity(source, target) > 0.3
