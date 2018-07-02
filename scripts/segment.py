'''
A python script to segment a text into sentences using the trained encoder and classifier
* Read the text to processed from a file
* Use segmenter class to segment text into sentences
* Print sentences on standard output
'''

import os
import re
import argparse
from configparser import SafeConfigParser

from srcs.Segmenter import Segmenter

conf = SafeConfigParser()
conf.read(os.path.join(os.path.dirname(__file__), '..', 'config', 'configuration.ini'))

eosPunctRE = re.compile(r'[\.?!:;]')

def loadText(isHtml: bool) -> str:
  '''Load input text from a file

  Args:
    isHtml: Enable html processing, retrieve html input if activated

  Returns:
    The input text
  '''
  datasetFolderPath = os.path.join(os.path.dirname(__file__), '..',
                                   conf.get('DATASET', 'datasetFolder'))
  if isHtml:
    textFile = os.path.join(datasetFolderPath, conf.get('DATASET', 'htmlInput'))
  else:
    textFile = os.path.join(datasetFolderPath, conf.get('DATASET', 'input'))
  content = ''
  with open(textFile, 'r') as inputFd:
    content = inputFd.read()
  return content

def main(args: argparse.Namespace) -> None:
  '''Main segmentation function to split the text into sentences or span tags

  Args:
    args: An argument Namespace
  '''
  text = loadText(args.html)
  segmenter = Segmenter(args.ngram_size, args.html)
  result = segmenter.segment(text, args.debug)

  if result and not args.debug:
    if not args.html:
      for sentence in result:
        print(sentence)
    else:
      print(result)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--ngram-size', default=5, type=int, help='Character ngram size')
  parser.add_argument('--html', default=False, action='store_const', const=True,
                      help='Enable the html processing of the text')
  parser.add_argument('--debug', default=False, action='store_const', const=True,
                      help='Enable prompt debugging of the html feature')
  arguments = parser.parse_args()
  main(arguments)
