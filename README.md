# sentence-boundary-detection
Detect sentence boundaries using machine learning

# Python version
This project was tested with python 3.5.2 using `pyenv`

**Add `$(pwd)/srcs/` in your environement variable `PYTHONPATH` otherwise some scripts won't run**

# Dependencies
Run the following
* `./install.sh`

# Check code quality with the linter
* `./linter.sh`

# Usage
You can use the option `-h` on each script to get help indications with the available options

## Get the dataset
(Around one 200Mo compressed, and about 600Mo uncompressed)
It can take up to 10 minutes to retrieve the dataset from the source (the remote server is slow)
* `python scripts/acquire.py`

## Preprocess the dataset
* `python scripts/preprocess.py --samples 10000000 [--samples NB_INSTANCES --ngram-size NGRAM_SIZE]`

## Train a model from the preprocessed dataset
* `python scripts/train.py [--balance --rnn --epoch NB_EPOCH]`

## Test the model after training it
* `python scripts/test.py [--balance --rnn]`

## Segment an input text into sentences
To modify the input text open the file located here "*datasets/input.txt*" and edit the input text
* `python scripts/segment.py [--ngram-size NGRAM_SIZE --html --debug]`

To modify the input HTML open the file located here "*datasets/input.html*" and edit the input HTML
* `python scripts/segment.py --html`

The manual way to check if tags were inserted at the right place (mosly the end span tag because the start span tag insertion is flawed) is to store the result in a file like that
* `python scripts/segment.py --html > result.html`

Then you can run the following line to print the lines in the html where span tags were inserted
* `python scripts/segment.py --html --debug`

# Tasks that were planned but not achieved

## Not achieved due to time
* Do some data science analysis by exploring the dataset, that would have helped to define custom features to add on top of TfIdf vectors. And better tune the number of ngrams to use.
* Write regression tests.
* Add an option to build a model with both the french and english datasets that I retrieve from `acquire.py` and test that model to see if it is better.

## Not achieved due to technical issues
* Build a decent model with an LSTM architecture, all models that I built were not decent due to the training time required to feed a nice portion of the dataset
* Due to the training time I didn't use matplotlib to plot training scores on the evaluation dataset.
* Hyperparameters optimization, it is not needed for a Naive Bayes nor a TfIdf because there is too few parameters to tune. However it would have been useful with a neural network.

# Ideas that will not be tested

* Tokenization to test the results with words instead of bag of characters
* If words were used, word embeddings from word2vec, GloVe or fastText could have been used
* Also if I were using words I could have tested the effect of part of speech tagging
* Pre-trained character embeddings could exist
* With an acronym detector, the classifier could perform better because it will generalize contexts
that contain acronyms (e.g: the dataset contains a lot of M. acronyms if all acronyms would be
replaced by a tag then the classification behavior will generalise for other acronyms that are way less present in the dataset such as
Mme. or Dr.)
