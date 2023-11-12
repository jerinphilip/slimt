# type: ignore
import os
from slimt.utils import toJSON
from slimt import REPOSITORY, Package, Model, preset
import yaml


def test_ende_borked_base(service, models, sample):
    model_id = "en-de-base"
    repository = "browsermt"
    REPOSITORY.download(repository, model_id)
    config_path = REPOSITORY.modelConfigPath(repository, model_id)
    c = None
    with open(config_path) as yaml_file:
        c = yaml.safe_load(yaml_file)

    root = os.path.dirname(config_path)
    package = Package(
        model=os.path.join(root, c["models"][0]),
        vocabulary=os.path.join(root, c["vocabs"][0]),
        shortlist=os.path.join(root, c["shortlist"][0]),
    )

    config = preset.base()
    config.decoder_layers = 1
    model = Model(config, package)

    source, _, html = sample
    responses = service.translate(model, [source], html=html)
    for response in responses:
        print(toJSON(response, indent=4))
