# type: ignore
from slimt.utils import toJSON


def test_basic(service, models, source_and_target):
    source, _, html = source_and_target
    model = models[0]
    responses = service.translate(model, [source], html=html)
    for response in responses:
        print(toJSON(response, indent=4))
