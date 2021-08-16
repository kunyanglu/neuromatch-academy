import time
# import fasttext
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import requests
import torch
import torch.nn as nn
from torch.nn import functional as F

from hmmlearn import hmm
from scipy.sparse import dok_matrix

from torchtext import data, datasets
from torchtext.vocab import FastText

# import nltk
from nltk import FreqDist
from nltk.corpus import brown
from nltk.tokenize import word_tokenize

from gensim.models import Word2Vec

from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm_notebook as tqdm
from helper import \
        cosine_similarity, tokenize, plot_train_val, \
        download_file_from_google_drive, get_confirm_token, \
        save_response_content
        
def load_dataset(emb_vectors, sentence_length=50, seed=522):
    TEXT = data.Field(sequential=True,
                    tokenize=tokenize,
                    lower=True,
                    include_lengths=True,
                    batch_first=True,
                    fix_length=sentence_length)
    LABEL = data.LabelField(dtype=torch.float)

    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

    TEXT.build_vocab(train_data, vectors=emb_vectors)
    LABEL.build_vocab(train_data)

    train_data, valid_data = train_data.split(split_ratio=0.7,
                                            random_state=random.seed(seed))
    train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data,
                                                                  valid_data,
                                                                  test_data),
                                                                  batch_size=32,
                                                                  sort_key=lambda x: len(x.text),
                                                                  repeat=False,
                                                                  shuffle=True)
    vocab_size = len(TEXT.vocab)

    print(f'Data are loaded. sentence length: {sentence_length} '
        f'seed: {seed}')

    return TEXT, vocab_size, train_iter, valid_iter, test_iter

def get_corpus(sentences):
    corpus_words = []
    for sentence in sentences:
        for word in sentence:
            if "''" not in word and "``" not in word:
                corpus_words.append(word)
    
    return corpus_words

def create_matrix_representation(corpus_words):
    # Prepare for filling the matrix later
    distinct_words = list(set(corpus_words))
    word_idx_dict = {word: i for i, word in enumerate(distinct_words)}
    distinct_words_count = len(list(set(corpus_words)))
    next_word_matrix = np.zeros([distinct_words_count, distinct_words_count])

    # Populate the matrix
    for i, word in enumerate(corpus_words[:-1]):
        first_word_idx = word_idx_dict[word]
        next_word_idx = word_idx_dict[corpus_words[i+1]]
        next_word_matrix[first_word_idx][next_word_idx] +=1

    return word_idx_dict, next_word_matrix, distinct_words

##################################
### Section 1.2 - Marcov Chain ###
##################################

def most_likely_word_after(word, word_idx_dict, next_word_matrix, 
                            distinct_words):
    # We check for the word most likely to occur using the matrix
    most_likely = next_word_matrix[word_idx_dict[word]].argmax()
    return distinct_words[most_likely]

def naive_chain(word, word_idx_dict, distinct_words, next_word_matrix, length=15):
    current_word = word
    sentence = word
    # We now build a naive chain by picking up the most likely word
    for _ in range(length):
        sentence += ' '
        next_word = most_likely_word_after(current_word, word_idx_dict, 
                                next_word_matrix, distinct_words)
        sentence += next_word
        current_word = next_word

    return sentence

def weighted_choice(objects, weights):
  """
  Returns randomly an element from the sequence of 'objects',
      the likelihood of the objects is weighted according
      to the sequence of 'weights', i.e. percentages.
  """
    weights = np.array(weights, dtype=np.float64)
    sum_of_weights = weights.sum()
    
    # standardization:
    np.multiply(weights, 1 / sum_of_weights)
    weights = weights.cumsum()
    x = random.random()
    
    for i in range(len(weights)):
        if x < weights[i]:
            return objects[i]

def sample_next_word_after(word, next_word_matrix, word_idx_dict, alpha=0):
    next_word_vector = next_word_matrix[word_idx_dict[word]] + alpha
    likelihoods = next_word_vector/next_word_vector.sum()
    return weighted_choice(distinct_words, likelihoods)

def stochastic_chain(word, next_word_matrix, word_idx_dict, length=15):
    current_word = word
    sentence = word

    for _ in range(length):
        sentence += ' '
        next_word = sample_next_word_after(current_word, 
                        next_word_matrix, word_idx_dict)
        sentence += next_word
        current_word = next_word

    return sentence

if __name__ == "__main__":

    category = ['editorial', 'fiction', 'government', 'news', 'religion']
    sentences = brown.sents(categories=category)
    
    corpus_words = get_corpus(sentences)
    word_idx_dict, next_word_matrix, distinct_words = create_matrix_representation(corpus_words)

    print(naive_chain('the', word_idx_dict, distinct_words, next_word_matrix))
    print(naive_chain('I', word_idx_dict, distinct_words, next_word_matrix))
    print(naive_chain('What', word_idx_dict, distinct_words,  next_word_matrix))
    print(naive_chain('park', word_idx_dict, distinct_words,  next_word_matrix))

    
