'''
A class to segment text into sentences
* Load encoder model
* Load classification model
* Write a custom processing function for this script inspired by what is done in `preprocess` script
* Classify each processed token
* With classification resultats split the text into predicted sentences
* Do the same pipeline for the html option
'''

__all__ = ['Segmenter']

from typing import List, Union, Tuple
import os
import math
import re
from configparser import SafeConfigParser
from html.parser import HTMLParser

from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

conf = SafeConfigParser()
conf.read(os.path.join(os.path.dirname(__file__), '..', 'config', 'configuration.ini'))

class TextIndexParser(HTMLParser):
  '''A class to parse HTML content, retrieve the inner content of tags

  Attributes:
    lines (list): A list of strings, each string is the inner content of a pair of tags
  '''
  def __init__(self, *args, **kwargs) -> None:
    self.lines = []
    super(TextIndexParser, self).__init__(*args, **kwargs)

  def handle_data(self, data: str) -> None:
    '''Function called at each tag, store each inner content with its start and end position

    Args:
      data: A tag inner content
    '''
    if data.strip():
      start = self.getpos()
      end = (start[0], start[1] + len(data))
      self.lines.append((data, start, end))

  def clear(self) -> None:
    '''Reset the internal storage of tags inner content
    '''
    self.lines = []

  def error(self) -> None:
    '''Overwritten because the linter would print errors, however the function is not documentated
    in the standard api, that's why it is empty
    '''
    pass

def yxToIndex(yIndex: int, xIndex: int, htmlLines: List[str]) -> int:
  '''Transform a (y, x) index into a flat index in the raw html content

  Args:
    yIndex:     A line index
    xIndex:     A character index within a line
    htmlLines:  Html content split by new line

  Returns:
    An index in the raw html content
  '''
  index = 0
  j = 0

  while j < yIndex:
    index += len(htmlLines[j]) + 1
    j += 1

  return index + xIndex

class Segmenter(object):
  '''A class to segment text or html into sentences

  Args:
    ngramSize:  The number of characters used for context tokens,
                should be the same number as for preprocessing
    isHtml:     Enable html preprocessing and segmentation

  Attributes:
    pipeline    (Pipeline):     A chaining object to chain scikit algorithms,
                                will contain an encoder and a classifier
    ngramSize   (int):          The number of characters for context tokens
    isHtml      (bool):         Trigger to enable html preprocessing and segmentation
    eosPunctRE  (SRE_Pattern):  A regex with end-of-sentence punctuation
  '''
  def __init__(self, ngramSize: int, isHtml: bool) -> None:
    folderPath = os.path.join(os.path.dirname(__file__), '..', conf.get('MODEL', 'modelFolder'))
    self.pipeline = Pipeline([
      ('encoder', joblib.load(os.path.join(folderPath, conf.get('MODEL', 'encoder')))),
      ('classifier', joblib.load(os.path.join(folderPath, conf.get('MODEL', 'classifier'))))
    ])
    self.ngramSize = ngramSize
    self.isHtml = isHtml
    self.eosPunctRE = re.compile(r'[\.?!:;]')
    self.removeHtmlIndexText = TextIndexParser()

  def __preprocess(self, text: str) -> List[Tuple[str, int]]:
    '''Preprocessing function, retrieve only tokens that are eligible as end of sentences and cut a
    window of characters around the end of sentence character

    Args:
      text:       A text input to split into sentences

    Returns:
      A list of tuples, each tuple has a token which has a potential end of sentence token
      and a window of characters around, moreover the tuple has the index next to the eos character
      to split sentences
    '''
    ngramWindow = math.floor(self.ngramSize / 2)
    tokens = []
    matches = re.finditer(self.eosPunctRE, text)

    for match in matches:
      token = text[match.start() - ngramWindow:match.end() + ngramWindow].replace('\n', ' ').lower()
      # custom right padding
      while len(token) < self.ngramSize:
        token = token + ' '
      if len(token) == self.ngramSize:
        tokens.append((token, match.end()))

    return tokens

  def segment(self, text: str, debug: bool) -> Union[List[str], str]:
    '''Takes a text as input, preprocess into to find end-of-sentence tokens in it, predict if
    the token is an end of sentence. Then split the text into sentences with the selected
    end-of-sentence tokens and their indexes

    Args:
      text:   Input text to segment
      debug:  Enables span insertion debugging

    Returns:
      A list of sentences or an html block with span tags for each sentence
    '''
    result = None

    if not self.isHtml:
      tokens = self.__preprocess(text)

      if not tokens:
        return result

      predictions = self.pipeline.predict(list(zip(*tokens))[0])

      result = []
      # split text into sentences
      index = 0
      for i, token in enumerate(tokens):
        if predictions[i] == 1:
          result.append(text[index:token[1]].strip())
          index = token[1]
      result.append(text[index:].strip())
    else:
      # Build a list of indexes to then insert span tags at these indexes
      result = text
      htmlByLine = text.split('\n')
      self.removeHtmlIndexText.feed(text)
      textIndexes = self.removeHtmlIndexText.lines
      self.removeHtmlIndexText.clear()

      offset = 0
      startSentence = 0

      for textIndex in textIndexes:
        tokens = self.__preprocess(textIndex[0])
        if not tokens:
          continue

        predictions = self.pipeline.predict(list(zip(*tokens))[0])

        for i, token in enumerate(tokens):
          if predictions[i] == 1:
            if debug:
              print('Line index: {}\nContent: {}'.format(textIndex[1][0], textIndex[0]))
            # Consider that the end of the previous sentence is the beginning of a new one, even if
            # this assumption is false in an html file
            result = result[:startSentence] + '<span>' + result[startSentence:]
            offset += len('<span>')
            # Find the index in the original html from the splitted html and the token classified
            xyIndex = yxToIndex(textIndex[1][0] - 1, textIndex[1][1] + token[1] - 1, htmlByLine)
            # Add the end of sentence span
            result = result[:xyIndex + offset + 1] + '</span>' + result[xyIndex + offset + 1:]
            offset += len('</span>')
            # Update the start of sentence index
            startSentence = xyIndex + offset + 1

    return result
