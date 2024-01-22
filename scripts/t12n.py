import yaml
import os
import slimt
import sys
from slimt import Model, Package, Config
from slimt import Service, Model
from slimt import preset


# Load the config file
def load_config(path):
    config = None
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


if __name__ == "__main__":
    service = Service(workers=1, cache_size=1024)
    # Load supplementary files for model execution by passing the config path directory
    config_path = sys.argv[1]
    root = os.path.dirname(config_path)
    config = load_config(config_path)
    package = Package(
        model=os.path.join(root, config["models"][0]),
        vocabulary=os.path.join(root, config["vocabs"][0]),
        shortlist=os.path.join(root, config["shortlist"][0]),
        # shortlist="",
    )

    # nano model
    nano: Config = preset.nano()
    model_nano = Model(nano, package)

    data = sys.stdin.read()
    responses = service.translate(model_nano, [data], html=False)

    for response in responses:
        print(response.source.text, "->", response.target.text)
