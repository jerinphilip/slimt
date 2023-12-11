from .words import WordIterator

class SentenceIterator:
  def __init__(self, annotation):
    self._annotation = annotation
    self._sentence_id = -1

  def __iter__(self):
    self._sentence_id = -1
    return self

  def __next__(self):
    self._sentence_id += 1
    if self._sentence_id >= self._annotation.sentence_count():
      raise StopIteration
    return self

  def words(self):
    return WordIterator(self._annotation, self._sentence_id)

  def __repr__(self):
    range = self._annotation.sentence_as_range(self._sentence_id)
    sentence = self._annotation.text[range.begin:range.end]
    return f'{sentence}'
