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

from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

conf = SafeConfigParser()
conf.read(os.path.join(os.path.dirname(__file__), '..', 'config', 'configuration.ini'))

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
      if len(token) == self.ngramSize:
        tokens.append((token, match.end()))

    return tokens

  def segment(self, text: str) -> Union[List[str], str]:
    '''Takes a text as input, preprocess into to find end-of-sentence tokens in it, predict if
    the token is an end of sentence. Then split the text into sentences with the selected
    end-of-sentence tokens and their indexes

    Args:
      text: Input text to segment

    Returns:
      A list of sentences or an html block with span tags for each sentence
    '''
    tokens = self.__preprocess(text)
    predictions = self.pipeline.predict(list(zip(*tokens))[0])
    result = None

    if not self.isHtml:
      result = []
      # split text into sentences
      index = 0
      for i, token in enumerate(tokens):
        if predictions[i] == 1:
          result.append(text[index:token[1]].strip())
          index = token[1]
      result.append(text[index:].strip())
    else:
      # add span tags around sentences in html content
      result = text
    return result
