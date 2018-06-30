'''
A python script to test a model trained with the `training.py` script.
* Load encoder model
* Load classification model
* Load testing dataset
* Predict each label for each testing instance
* Use the predictions and the testing labels to compute usual classification metrics
  such as precision, recall, f1_score
'''

from typing import Tuple
import os
import argparse
from configparser import SafeConfigParser
from pprint import pprint

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.externals import joblib

conf = SafeConfigParser()
conf.read(os.path.join(os.path.dirname(__file__), '..', 'config', 'configuration.ini'))

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

def loadDataset() -> pd.DataFrame:
  '''Load testing dataset into a DataFrame from file

  Returns:
    A dataframe that contain the testing dataset
  '''
  return pd.read_json(os.path.join(os.path.dirname(__file__),
                                   '..',
                                   conf.get('DATASET', 'datasetFolder'),
                                   conf.get('DATASET', 'test')),
                      orient='records', typ='frame')

def main(args: argparse.Namespace) -> None:
  '''Main testing function to evaluate the model trained on our dataset

  Args:
    args: An argument Namespace
  '''
  encoder, classifier = loadModels()
  pipeline = Pipeline([('encoder', encoder), ('classifier', classifier)])
  dataset = loadDataset()

  eosInstances = dataset[dataset['label'] == 1.0]
  neosInstances = dataset[dataset['label'] == 0.0]

  print('Number of instances as end of sentence:\t\t', len(eosInstances))
  print('Number of instances as not end of sentence:\t', len(neosInstances))

  if args.balance:
    dataset = pd.concat([neosInstances, eosInstances[:len(neosInstances)]])

  predictions = pipeline.predict(dataset['instance'])
  report = classification_report(dataset['label'], predictions)
  print(report)

  confusionMatrix = confusion_matrix(dataset['label'], predictions).ravel()
  instancesMatrix = dict(zip(('tn', 'fp', 'fn', 'tp'), confusionMatrix))
  pprint(instancesMatrix)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--balance', default=False, action='store_const', const=True,
                      help='Use the same number of instances per label value count')
  arguments = parser.parse_args()
  parser.print_help()
  main(arguments)
