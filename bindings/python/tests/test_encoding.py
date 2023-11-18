# type: ignore
from slimt.utils import toJSON
from slimt import Encoding


def test_basic(service, models, sample):
    source = "no sé"
    model = models[1]
    responses_byte = service.translate(
        model, [source], html=False, encoding=Encoding.Byte
    )
    responses_utf8 = service.translate(
        model, [source], html=False, encoding=Encoding.UTF8
    )
    for response in responses_byte:
        print(dir(response))
        print(toJSON(response, indent=4))
        extracted_byte = response.source.word(0, 0)
    for response in responses_utf8:
        print(dir(response))
        print(toJSON(response, indent=4))
        range = response.source.word_as_range(0, 0)
        extracted_utf8 = response.source.text[range.begin : range.end]
    assert extracted_utf8 == extracted_byte
