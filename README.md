# sentence-boundary-detection
Detect sentence boundaries using machine learning

# Python version
This project was tested with python 3.5.2 using `pyenv`

# Usage
You can use the option `-h` on each script to get help indications with the available options

## Get the dataset
* `python scripts/acquire.py`

## Preprocess the dataset
* `python scripts/preprocess.py`

## Train a model from the preprocessed dataset
* `python scripts/train.py`

## Test the model after training it
* `python scripts/test.py`

## Segment an input text into sentences
To modify the input text open the file located here "*datasets/input.txt*" and edit the input text
* `python scripts/segment.py`

To modify the input HTML open the file located here "*datasets/input.html*" and edit the input HTML
* `python scripts/segment.py --html`

The manual way to check if tags were inserted at the right place (mosly the end span tag because the start span tag insertion is flawed) is to store the result in a file like that
* `python scripts/segment.py --html > result.html`

Then you can run the following line to print the lines in the html where span tags were inserted
* `python scripts/segment.py --html --debug`

# Ideas that will not be tested
* Tokenization to test the results with words instead of bag of characters
* If words were used, word embeddings from word2vec, GloVe or fastText could have been used
* Also if I were using words I could have tested the effect of part of speech tagging
* Pre-trained character embeddings could exist
* With an acronym detector, the classifier could perform better because it will generalize contexts
that contain acronyms (e.g: the dataset contains a lot of M. acronyms if all acronyms would be
replaced by a tag then the classification behavior will generalise for other acronyms that are way less present in the dataset such as
Mme. or Dr.)
