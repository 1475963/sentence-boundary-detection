'''
A python script to segment a text into sentences using the trained encoder and classifier
* Read the text to processed from a file
* Use segmenter class to segment text into sentences
* Print sentences on standard output
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

from srcs.Segmenter import Segmenter

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

def main(args: argparse.Namespace) -> None:
  '''Main segmentation function to split the text into sentences or span tags

  Args:
    args: An argument Namespace
  '''
  text = loadText()
  segmenter = Segmenter(args.ngram_size, args.html)
  sentences = segmenter.segment(text)

  for sentence in sentences:
    print(sentence)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--ngram-size', default=5, type=int, help='Character ngram size')
  parser.add_argument('--html', default=False, action='store_const', const=True,
                      help='Enable the html processing of the text')
  arguments = parser.parse_args()
  main(arguments)
