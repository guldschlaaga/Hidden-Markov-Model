from time import time
import numpy as np
import os
import sys


# File: nlputil.py
# Purpose: A collection of utility methods for working with text data.
# Author: Adam Purtee (starter code for CSC 246 Spring 2021 third Project)
#
#
# This file contains three kinds of methods for each of the following tasks:
#  o building a vocabulary s-- this obtains a dictionary from tokens to integers
# #  o converting to integers -- this converts a sample to a sequence of integers
# #  o converting to one-hot -- this converts a sample to a sequence of one-hot vectors
# #                             (i.e., this maps strings to matrices).
# #
# # There are parallel methods included for working with either word-based or character-based models.
# # In both cases, unknown words will always map to integer value 0.
#
# # Note -- File IO is slow, so it's best if you can keep as much data in RAM as possible.
# # Converting to a word-based one-hot representation is EXTREMELY memory intensive (many huge vectors of ints).
# # Converting to a character-based one-hot representation is also expensive (more vectors, but smaller dimension).
# # The most memory efficient implementation would probably read all of the data into RAM, leave it as a gigantic
# # python list of strings, and do vocabulary lookup on the fly during learning/inference.
# # The fastest implementation would precompute as much stuff as possible.
#
#
# # A vocabulary for this project is a dictionary from tokens to unique integers.
# #
# # This will simply assign each unique word in the data a unique integer, starting with one
# # and increasing until all word are processed.  The exact results depend on the order in
# which the paths are presented to the method.  Renaming the files may also change the order.
#
# paths -- a list of paths to directories containing data.  paths represented as strings)
def build_vocab_words(paths):
    vocab = {}
    nextValue = 1
    for path in paths:
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), "rb") as fh:
                sequence = fh.read()
                for token in sequence.split():
                    if token not in vocab:
                        vocab[token] = nextValue
                        nextValue += 1
    return vocab


# Same as above, but for character models.
def build_vocab_chars(paths):
    vocab = {}
    nextValue = 1
    for path in paths:
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), "rb") as fh:
                sequence = fh.read()
                for character in sequence:
                    if character not in vocab:
                        vocab[character] = nextValue
                        nextValue += 1
    return vocab


# Sample is a plain string - not a list -- UNK token has value zero.
# Convert the sample to a integer representation, which is an Nx1 array of ints,
# where N is the number of tokens in the sequence.
def convert_words_to_ints(sample, vocab):
    sequence = sample.split()
    answer = np.zeros(len(sequence), dtype=np.uint)
    for n, token in enumerate(sequence):
        answer[n] = vocab.get(token, 0)
    return answer


# Same as above, but for characters.
def convert_chars_to_ints(sample, vocab):
    answer = np.zeros(len(sample), dtype=np.uint)
    for n, token in enumerate(sample):
        answer[n] = vocab.get(token, 0)
    return answer


# Sample is a plain string - not a list -- UNK token has value zero.
# Convert the sample to a one-hot representation, which is an NxV matrix,
# where N is the number of tokens in the sequence and V is the vocabulary
# size observed on the training data.
def convert_words_to_onehot(sample, vocab):
    sequence = sample.split()
    onehot = np.zeros((len(sequence), len(vocab) + 1), dtype=np.uint)
    for n, token in enumerate(sequence):
        onehot[n, vocab.get(token, 0)] = 1
    return onehot


# Same as above, but for characters.
def convert_chars_to_onehot(sample, vocab):
    onehot = np.zeros((len(sample), len(vocab) + 1), dtype=np.uint)
    for n, token in enumerate(sample):
        onehot[n, vocab.get(token, 0)] = 1
    return onehot


# Read every file located at given path, convert to one-hot OR integer representation,
# and collect the results into a python list. 
def load_and_convert_data_words_to_onehot(paths, vocab):
    data = []
    for path in paths:
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), "rb") as fh:
                data.append(convert_words_to_onehot(fh.read(), vocab))
    return data


# Same as above, but uses a character model
def load_and_convert_data_chars_to_onehot(paths, vocab):
    data = []
    for path in paths:
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), "rb") as fh:
                data.append(convert_chars_to_onehot(fh.read(), vocab))
    return data


def load_and_convert_data_words_to_ints(paths, vocab):
    data = []
    for path in paths:
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), "rb") as fh:
                data.append(convert_words_to_ints(fh.read(), vocab))
    return data


# Same as above, but uses a character model
def load_and_convert_data_chars_to_ints(paths, vocab):
    data = []
    for path in paths:
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), "rb") as fh:
                data.append(convert_chars_to_ints(fh.read(), vocab))
    return data


if __name__ == '__main__':
    print("NLP Util smoketest.");
    paths = [r"imdbFor246\train\pos",
             r"imdbFor246\train\neg"]
    print("Begin loading vocab... ", end='')
    sys.stdout.flush()
    begin = time()
    vocab = build_vocab_words(paths)
    end = time()
    print('done in', end - begin, 'seconds.  Found', len(vocab), 'unique tokens.')
    print('Begin loading all data and converting to ints... ', end='')
    sys.stdout.flush()
    begin = time()
    data = load_and_convert_data_words_to_ints(paths, vocab)
    end = time()
    print('done in', end - begin, 'seconds.')

    print("Data[0] = ", data[0])
    print('Press enter to quit.')
    input()
    print('Quitting.. may take some time to free memory.')
