class WordIterator:
  def __init__(self, annotation, sentence_id=None):
    self._annotation = annotation
    
    if sentence_id == None:
      self._sentence_id = 0
      self._max_sentence_id = self._annotation.sentence_count()
    else:
      self._sentence_id = sentence_id
      self._max_sentence_id = sentence_id + 1

    self._word_id = -1

  def __iter__(self):
    self._word_id = -1
    return self

  def __next__(self):
    if self._annotation.sentence_count() == 0:
      raise StopIteration

    self._word_id += 1
    if self._word_id >= self._annotation.word_count(self._sentence_id):
      self._sentence_id += 1
      if self._sentence_id >= self._max_sentence_id:
        raise StopIteration
      self._word_id = 0
    return self

  def surface(self):
    range = self.range()
    return self._annotation.text[range.begin:range.end]

  def range(self):
    return self._annotation.word_as_range(self._sentence_id, self._word_id)

  def id(self):
    return (self._sentence_id, self._word_id)

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
  
def sentences(annotation):
  return SentenceIterator(annotation)

def words(annotation, sentence_id=None):
  return WordIterator(annotation, sentence_id)
