'''
A python script to preprocess the sentence dataset, it will
* Select a specific number of instances from the full dataset
* Preprocess each sentence by cleaning and standardizing
* Transform each sentence into a bag of character ngrams
* Generate a training, a validation and a testing set
'''

import os
import re
from configparser import SafeConfigParser
from typing import List, Dict, Union
import argparse
import math
import json

conf = SafeConfigParser()
conf.read(os.path.join(os.path.dirname(__file__), '..', 'config', 'configuration.ini'))

eosPunctRE = re.compile(r'[\.?!:;]')
eosPunctDuplicateRE = re.compile(r'[\.?!:;]{2,}')

def sample(limit: int) -> List[str]:
  '''A function to load the original dataset and keep only a portion of it

  Args:
    limit:  The number of instances (sentences) to keep

  Returns:
    A list of sentences
  '''
  datasetFolderPath = os.path.join(os.path.dirname(__file__), '..',
                                   conf.get('DATASET', 'datasetFolder'))
  dataset = []
  with open(os.path.join(datasetFolderPath, conf.get('DATASET', 'fr-FR')), 'r') as datasetFd:
    for line in datasetFd:
      if len(dataset) >= limit:
        return dataset
      line = line.replace('\n', '')
      if line:
        dataset.append(line)
  return dataset

def preprocess(dataset: List[str]) -> List[str]:
  '''Preprocess each sentence by removing duplicate eos punctuations, set sentence to lowercase.
  It also removes sentences that do not end with an end of sentence punctuation

  Args:
    dataset:      A dataset of sentences

  Returns:
    A list of preprocessed sentences
  '''
  return [
    re.sub(eosPunctDuplicateRE, lambda m: ''.join(list(set(list(m.group())))), instance.lower())
    for instance in dataset
    if re.match(eosPunctRE, instance[-1])
  ]

def transform(dataset: List[str], ngramSize: int) -> List[Dict[str, Union[float, str]]]:
  '''Transform each sentence to a list of end of sentence ngrams

  Args:
    dataset:    A dataset of sentences
    ngramSize:  Ngram size

  Returns:
    A list of end of sentence ngrams with their associated label
  '''
  ngramWindow = math.floor(ngramSize / 2)
  dataset = '\n'.join(dataset)

  return [
    {
      'label': 1.0,
      'instance': dataset[m.start() - ngramWindow:m.end() + ngramWindow].replace('\n', ' ')
    }
    if m.end() < len(dataset) and dataset[m.end()] == '\n'
    else
    {
      'label': 0.0,
      'instance': dataset[m.start() - ngramWindow:m.end() + ngramWindow].replace('\n', ' ')
    }
    for m in re.finditer(eosPunctRE, dataset)
  ][:-1]

def split(dataset: List[Dict[str, Union[float, str]]]) -> Dict[str, List[dict]]:
  '''Split the dataset to form a training set, a validation set and a testing set

  Args:
    dataset:  A dataset of sentences

  Returns:
    A dict of datasets
  '''
  proportions = {'train': .7, 'validation': .1, 'test': .2}
  cuts = {setName:round(len(dataset) * proportion) for setName, proportion in proportions.items()}

  return {
    'train': dataset[:cuts['train']],
    'validation': dataset[cuts['train']:cuts['train'] + cuts['validation']],
    'test': dataset[
      cuts['train'] + cuts['validation']:cuts['train'] + cuts['validation'] + cuts['test']]
  }

def save(sets: Dict[str, List[dict]]) -> None:
  '''A function to save splitted datasets into disk

  Args:
    sets: A dict of datasets
  '''
  for key in sets:
    with open(os.path.join(conf.get('DATASET', 'datasetFolder'),
                           conf.get('DATASET', key)), 'w') as fileDescriptor:
      json.dump(sets[key], fileDescriptor)

def main(args: argparse.Namespace) -> None:
  '''Main processing function to preprocess our dataset

  Args:
    args: An argument Namespace
  '''
  # retrieve few samples from the original dataset
  dataset = sample(args.samples)
  # remove duplicates
  dataset = list(set(dataset))
  print('dataset length:', len(dataset))
  # apply preprocess pipeline to sampled dataset
  dataset = preprocess(dataset)

  # transform the dataset, instead of a sentence per instance it will be a character ngram
  # per instance with its associated label, 1.0 for an end of sentence ngram and 0.0 other ngrams
  dataset = transform(dataset, args.ngram_size)

  sets = split(dataset)
  print('train:', len(sets['train']))
  print('validation:', len(sets['validation']))
  print('test:', len(sets['test']))

  save(sets)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--samples', default=100000, type=int,
                      help='Number of sentences to use from the original dataset')
  parser.add_argument('--ngram-size', default=5, type=int, help='Character ngram size')
  arguments = parser.parse_args()
  parser.print_help()
  main(arguments)
