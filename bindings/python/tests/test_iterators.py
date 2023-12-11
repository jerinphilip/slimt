# type: ignore
from slimt import iterators

def test_iterators(service, models):
    source = "Hi, How are you? Its been a long time.\nCan you help me out with some things?"
    model = models[1]
    response = service.translate(model, [source], html=False)[0]

    target = response.target
    text = target.text

    sen_iter_tgt = iterators.sentences(target)
    word_iter_global = iterators.words(target)

    sentence_count = target.sentence_count()
    for sentence_idx, word_iter in zip(range(sentence_count), sen_iter_tgt):
        word_count = target.word_count(sentence_idx)
        for word_idx, word in zip(range(word_count), word_iter.words()):
            
            expected_text_range = target.word_as_range(sentence_idx, word_idx)
            reconstructed_text_range = word.range()

            # For Sentence Iterator and Word Iterator
            assert expected_text_range.begin == reconstructed_text_range.begin
            assert expected_text_range.end == reconstructed_text_range.end

            expected_word = text[expected_text_range.begin:expected_text_range.end]
            reconstructed_word = word.surface()

            assert expected_word == reconstructed_word

            word_global = next(word_iter_global)

            reconstructed_text_range_glob = word_global.range()
            reconstructed_word_glob = word_global.surface()

            # For Global Word Iterator
            assert expected_text_range.begin == reconstructed_text_range_glob.begin
            assert expected_text_range.end == reconstructed_text_range_glob.end
            assert expected_word == reconstructed_word_glob