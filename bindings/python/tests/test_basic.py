# type: ignore
from slimt.utils import toJSON


def test_basic(service, models, sample):
    source, _, html = sample
    model = models[0]
    responses = service.translate(model, [source], html=html)
    for response in responses:
        print(toJSON(response, indent=4))
