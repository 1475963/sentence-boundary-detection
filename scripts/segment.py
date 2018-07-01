'''
A python script to segment a text into sentences using the trained encoder and classifier
* Read the text to processed from a file
* Load encoder model
* Load classification model
* Write a custom processing function for this script inspired by what is done in `preprocess` script
* Classify each processed token
* With classification resultats split the text into predicted sentences
* Do the same pipeline for the html option
'''

from typing import Tuple, List
import os
import re
import argparse
import math
from configparser import SafeConfigParser

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

conf = SafeConfigParser()
conf.read(os.path.join(os.path.dirname(__file__), '..', 'config', 'configuration.ini'))

eosPunctRE = re.compile(r'[\.?!:;]')

def loadText() -> str:
  '''Load input text from a file

  Returns:
    The input text
  '''
  datasetFolderPath = os.path.join(os.path.dirname(__file__), '..',
                                   conf.get('DATASET', 'datasetFolder'))
  content = ''
  with open(os.path.join(datasetFolderPath, conf.get('DATASET', 'input')), 'r') as inputFd:
    content = inputFd.read()
  return content

def loadModels() -> Tuple[TfidfVectorizer, MultinomialNB]:
  '''Load the encoder model and the classifier model

  Returns:
    An encoder model object
    A classifier model object
  '''
  folderPath = os.path.join(os.path.dirname(__file__), '..', conf.get('MODEL', 'modelFolder'))

  return (
    joblib.load(os.path.join(folderPath, conf.get('MODEL', 'encoder'))),
    joblib.load(os.path.join(folderPath, conf.get('MODEL', 'classifier')))
  )

def preprocess(text: str, ngramSize: int) -> List[str]:
  '''Preprocessing function, retrieve only tokens that are eligible as end of sentences and cut a
  window of characters around the end of sentence character

  Args:
    text:       A text input to split into sentences
    ngramSize:  Character ngram size

  Returns:
    A list of tuples, each tuple has a token which has a potential end of sentence token
    and a window of characters around, moreover the tuple has the index next to the eos character
    to split sentences
  '''
  ngramWindow = math.floor(ngramSize / 2)
  tokens = []
  match = re.finditer(eosPunctRE, text)

  for m in match:
    print('m:', m)
    token = text[m.start() - ngramWindow:m.end() + ngramWindow].replace('\n', ' ').lower()
    if len(token) == ngramSize:
      tokens.append((token, m.end()))

  return tokens

def main(args: argparse.Namespace) -> None:
  '''Main segmentation function to split the text into sentences or span tags

  Args:
    args: An argument Namespace
  '''
  encoder, classifier = loadModels()
  pipeline = Pipeline([('encoder', encoder), ('classifier', classifier)])
  text = loadText()

  print('=== INPUT TEXT ===')
  print(text)

  # preprocess and transform text
  tokens = preprocess(text, args.ngram_size)

  # classify tokens
  predictions = pipeline.predict(list(zip(*tokens))[0])

  if not args.html:
    # split text into sentences
    index = 0
    for i, token in enumerate(tokens):
      print('token:', token)
      print('pred:', predictions[i])
      if predictions[i] == 1:
        print('=== OUTPUT SENTENCE ===')
        print(text[index:token[1]].strip())
        index = token[1]
    print('=== OUTPUT SENTENCE ===')
    print(text[index:].strip())
  else:
    # split html into span tags
    pass

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--ngram-size', default=5, type=int, help='Character ngram size')
  parser.add_argument('--html', default=False, action='store_const', const=True,
                      help='Enable the html processing of the text')
  arguments = parser.parse_args()
  parser.print_help()
  main(arguments)
