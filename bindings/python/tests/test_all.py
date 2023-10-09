# type: ignore
import pytest
import yaml
import os
from slimt import REPOSITORY, Model, Service, Package, Config
from slimt.utils import toJSON


def test_basic():
    keys = ["browsermt"]
    models = ["de-en-tiny"]
    service = Service(workers=1, cache_size=1024)
    for repository in keys:
        # models = REPOSITORY.models(repository, filter_downloaded=False)
        for model in models:
            REPOSITORY.download(repository, model)

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
            source = "1 2 3 4 5 6 7 8 9"
            responses = service.translate(model, [source], html=False)
            for response in responses:
                print(toJSON(response, indent=4))
