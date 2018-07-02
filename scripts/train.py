'''
A python script to train a machine learning algorithm on a character based ngrams dataset,
each instance being labelised as being a sentence boundary or not
* Load dataset
* Encode dataset
* Train algorithm
* Save algorithm
'''

import os
import argparse
from configparser import SafeConfigParser

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

conf = SafeConfigParser()
conf.read(os.path.join(os.path.dirname(__file__), '..', 'config', 'configuration.ini'))

def loadDataset() -> pd.DataFrame:
  '''Load training dataset into a DataFrame from file

  Returns:
    A dataframe that contain the training dataset
  '''
  return pd.read_json(os.path.join(os.path.dirname(__file__),
                                   '..',
                                   conf.get('DATASET', 'datasetFolder'),
                                   conf.get('DATASET', 'train')),
                      orient='records', typ='frame')

def buildPipeline() -> Pipeline:
  '''Create a pipeline with encoding and classification

  Returns:
    A pipeline object that encodes the dataset then train a classifier
  '''
  return Pipeline([
    ('encoder', TfidfVectorizer(
      analyzer='char', max_df=0.7, min_df=1,
      ngram_range=(1, 5), use_idf=True)),
    ('classifier', MultinomialNB())
  ])

def saveModels(pipeline: Pipeline) -> None:
  '''Save the encoder model and the classifier model

  Args:
    pipeline: A pipeline object that contain the encoder and the classifier
  '''
  folderPath = os.path.join(os.path.dirname(__file__), '..', conf.get('MODEL', 'modelFolder'))

  if not os.path.exists(folderPath):
    os.makedirs(folderPath)

  joblib.dump(pipeline.named_steps['encoder'],
              os.path.join(folderPath, conf.get('MODEL', 'encoder')))
  joblib.dump(pipeline.named_steps['classifier'],
              os.path.join(folderPath, conf.get('MODEL', 'classifier')))

def main(args: argparse.Namespace) -> None:
  '''Main training function to train an algorithm on our training dataset

  Args:
    args: An argument Namespace
  '''
  dataset = loadDataset()
  pipeline = buildPipeline()

  eosInstances = dataset[dataset['label'] == 1.0]
  neosInstances = dataset[dataset['label'] == 0.0]

  print('Number of instances as end of sentence:\t\t', len(eosInstances))
  print('Number of instances as not end of sentence:\t', len(neosInstances))

  if args.balance:
    dataset = pd.concat([neosInstances, eosInstances[:len(neosInstances)]])

  pipeline.fit(dataset['instance'].tolist(), dataset['label'].tolist())

  saveModels(pipeline)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--balance', default=False, action='store_const', const=True,
                      help='Use the same number of instances per label value count')
  arguments = parser.parse_args()
  parser.print_help()
  main(arguments)
