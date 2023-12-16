# type: ignore
from slimt import Encoding
from slimt.utils import to_json
from collections import namedtuple


def test_encoding(service, models):
    Pair = namedtuple("Pair", ["byte", "utf8"])
    source = "no sé 😀 😃 😄 😁 😆 ⛄ 🤔"
    model = models[1]
    response_byte = service.translate(
        model, [source], html=False, encoding=Encoding.Byte
    )[0]
    response_utf8 = service.translate(
        model, [source], html=False, encoding=Encoding.UTF8
    )[0]
    print(to_json(response_byte))
    print(to_json(response_utf8))
    utf8: AnnotatedText = response_utf8.source
    byte: AnnotatedText = response_byte.source
    text = Pair(
        byte=utf8.text.encode(),
        utf8=utf8.text,
    )
    sentence_count = byte.sentence_count()
    for sentence_idx in range(sentence_count):
        word_count = byte.word_count(sentence_idx)
        for word_idx in range(word_count):
            text_range = Pair(
                byte=byte.word_as_range(sentence_idx, word_idx),
                utf8=utf8.word_as_range(sentence_idx, word_idx),
            )
            expected = text.utf8[text_range.utf8.begin : text_range.utf8.end]
            reconstructed = text.byte[
                text_range.byte.begin : text_range.byte.end
            ].decode("utf-8")

            assert expected == reconstructed

    response_utf8.to(Encoding.Byte)
    utf8_to_byte: AnnotatedText = response_utf8.source
    sentence_count = byte.sentence_count()
    for sentence_idx in range(sentence_count):
        word_count = byte.word_count(sentence_idx)
        for word_idx in range(word_count):
            byte_range = byte.word_as_range(sentence_idx, word_idx)
            utf8_to_byte_range = utf8_to_byte.word_as_range(sentence_idx, word_idx)
            assert byte_range.begin == utf8_to_byte_range.begin
            assert byte_range.end == utf8_to_byte_range.end
