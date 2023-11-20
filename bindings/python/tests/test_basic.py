# type: ignore
from slimt.utils import to_json


def test_basic(service, models, sample):
    source, _, html = sample
    model = models[0]
    responses = service.translate(model, [source], html=html)
    for response in responses:
        print(to_json(response, indent=4))
