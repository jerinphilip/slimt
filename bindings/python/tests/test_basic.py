# type: ignore
from slimt.utils import toJSON


def test_basic(service_instance, model_instances):
    service = service_instance
    source = "1 2 3 4 5 6 7 8 9"
    model = model_instances[0]
    responses = service.translate(model, [source], html=False)
    for response in responses:
        print(toJSON(response, indent=4))
