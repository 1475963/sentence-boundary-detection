# sentence-boundary-detection
Detect sentence boundaries using machine learning

# Python version
This project was tested with python 3.5.2 using `pyenv`

# Usage

### Get the dataset
* `python scripts/acquire.py`

### Preprocess the dataset
* `python scripts/preprocess.py`

### Train a model from the preprocessed dataset
* `python scripts/train.py`

### Test the model after training it
* `python scripts/test.py`

# Ideas that will not be tested
* Tokenization to test the results with words instead of bag of characters
* If words were used, word embeddings from word2vec, GloVe or fastText could have been used
* Also if I were using words I could have tested the effect of part of speech tagging
* Pre-trained character embeddings could exist
