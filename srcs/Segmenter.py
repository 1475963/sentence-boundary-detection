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

class Segmenter():
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
    match = re.finditer(self.eosPunctRE, text)

    for m in match:
      token = text[m.start() - ngramWindow:m.end() + ngramWindow].replace('\n', ' ').lower()
      if len(token) == self.ngramSize:
        tokens.append((token, m.end()))

    return tokens

  def segment(self, text: str) -> Union[List[str], str]:
    '''
    '''
    tokens = self.__preprocess(text)
    predictions = self.pipeline.predict(list(zip(*tokens))[0])

    if not self.isHtml:
      sentences = []
      # split text into sentences
      index = 0
      for i, token in enumerate(tokens):
        if predictions[i] == 1:
          sentences.append(text[index:token[1]].strip())
          index = token[1]
      sentences.append(text[index:].strip())
      return sentences
    else:
      # add span tags around sentences in html content
      html = ''
      return html
