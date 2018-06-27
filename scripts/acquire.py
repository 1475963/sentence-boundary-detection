'''
A python script to retrieve a sentence dataset and store it in filesystem
'''

import os
from configparser import SafeConfigParser
from urllib import request
import tarfile

conf = SafeConfigParser()
conf.read(os.path.join(os.path.dirname(__file__), '..', 'config', 'configuration.ini'))

def main() -> None:
  '''Main processing function to retrieve our dataset
  '''
  print('-> getting dataset from URL ({}) ...'.format(conf.get('ACQUIRE', 'datasetUrl')), end='')
  request.urlretrieve(conf.get('ACQUIRE', 'datasetUrl'), filename=conf.get('ACQUIRE', 'datasetZip'))
  print('\r-> getting dataset from URL ({}) [OK]'.format(conf.get('ACQUIRE', 'datasetUrl')))
  print('-> unzipping ...', end='')
  zipFd = tarfile.open(conf.get('ACQUIRE', 'datasetZip'), "r:gz")
  zipFd.extractall(conf.get('ACQUIRE', 'datasetFolder'))
  zipFd.close()
  print('\r-> unzipping [OK]')
  print('-> removing compressed file ...', end='')
  os.remove(conf.get('ACQUIRE', 'datasetZip'))
  print('\r-> removing compressed file [OK]')

if __name__ == '__main__':
  main()
