# type: ignore
import pytest
import yaml
import os
from slimt import REPOSITORY, Model, Service, Package, Config


@pytest.fixture(scope="session")
def service():
    service_ = Service(workers=1, cache_size=1024)
    return service_


@pytest.fixture(scope="session")
def models():
    keys = ["browsermt"]
    model_ids = ["en-es-tiny", "es-en-tiny"]
    models_ = []

    for repository in keys:
        for model_id in model_ids:
            REPOSITORY.download(repository, model_id)

    for repository in keys:
        for modelId in model_ids:
            config_path = REPOSITORY.model_config_path(repository, modelId)
            c = None
            with open(config_path) as yaml_file:
                c = yaml.safe_load(yaml_file)
            root = os.path.dirname(config_path)
            package = Package(
                model=os.path.join(root, c["models"][0]),
                vocabulary=os.path.join(root, c["vocabs"][0]),
                shortlist=os.path.join(root, c["shortlist"][0]),
            )
            config = Config()
            model = Model(config, package)
            models_.append(model)
    yield models_


@pytest.fixture(
    scope="session",
    params=[
        ("1 2 3 4 5 6 7 8 9", "1 2 3 4 5 6 7 8 9", False),
        (
            "How embarrassing. A fridge full of condiments and no food.",
            "Que embarazoso. Una nevera llena de condimentos y nada de comida.",
            False,
        ),
        ("<div>Hello</div>", "<div>Hola</div>", True),
    ],
)
def sample(request):
    yield request.param
