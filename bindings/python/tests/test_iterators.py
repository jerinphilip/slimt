# type: ignore
from slimt import iterators

def test_iterators(service, models):
    source = "Hi, How are you? Its been a long time.\nCan you help me out with some things?"
    model = models[1]
    response = service.translate(model, [source], html=False)[0]

    target = response.target
    text = target.text

    sentences = iterators.sentences(target)
    words = iterators.words(target)

    sentence_count = target.sentence_count()
    for sentence_idx, word_iter in zip(range(sentence_count), sentences):
        word_count = target.word_count(sentence_idx)
        for word_idx, word in zip(range(word_count), word_iter.words()):
            
            expected_range = target.word_as_range(sentence_idx, word_idx)  
            expected_word = text[expected_range.begin:expected_range.end]

            # For Sentence Iterator and Word Iterator
            # Range
            reconstructed = word.range()

            assert expected_range.begin == reconstructed.begin
            assert expected_range.end == reconstructed.end

            # Word
            reconstructed = word.surface()
            assert expected_word == reconstructed

            # For Global Word Iterator
            word_global = next(words)
            
            # Range
            reconstructed = word_global.range()
            assert expected_range.begin == reconstructed.begin
            assert expected_range.end == reconstructed.end

            # Word
            reconstructed = word_global.surface()
            assert expected_word == reconstructed
