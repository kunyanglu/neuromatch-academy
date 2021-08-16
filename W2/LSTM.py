import math
import time
import nltk
import random
import collections
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchtext import data, datasets
from d2l import torch as d2l
from helper import set_seed, seed_worker
from rnn import load_dataset, train, test

SEED = 2021
DEVICE = 'cpu'

class LSTM(nn.Module):
    
    def __init__(self, layers, output_size, hidden_size, vocab_size,
                    embed_size, device):
        super(LSTM, self).__init__()
        self.n_layers = layers
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.device = device
        
        self.word_embeddings = nn.Embedding(embed_size, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(embed_size, hidden_size, self.n_layers)
        self.fc = nn.Linear(self.n_layers*hidden_size, output_size)

    def forward(self, input_sentences):
        input = self.word_embeddings(input_sentences).permute(1, 0, 2)
        hidden = (torch.randn(self.n_layers, input.shape[1],
                              self.hidden_size).to(self.device),
                  torch.randn(self.n_layers, input.shape[1],
                              self.hidden_size).to(self.device))

        input = self.dropout(input)
        output, hidden = self.LSTM(input)

        h_n = hidden[0].permute(1, 0, 2)
        h_n = h_n.contiguous().view(h_n.shape[0], -1)

        logits = self.fc(h_n)

        return logits

class biLSTM(nn.Module):

    def __init__(self, output_size, hidden_size, vocab_size,
                    embed_size, device):
        super(biLSTM, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.device = device

        self.word_embeddings = nn.Embedding(embed_size, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.bilstm = nn.LSTM(embed_size, hidden_size, num_layers=2,
                                bidirectional=True)
        self.fc = nn.Linear(4*hidden_size, output_size)

    def forward(self, input_sentences):
        input = self.word_embeddings(input_sentences).permute(1, 0, 2)
        hidden = (torch.randn(4, input.shape[1], self.hidden_size).to(self.device),
                torch.randn(4, input.shape[1], self.hidden_size).to(self.device))

        input = self.dropout(input)
        output, hidden = self.bilstm(input, hidden)
        h_n = hidden[0].permute(1, 0, 2)
        h_n = h_n.contiguous().view(h_n.shape[0], -1)
        logits = self.fc(h_n)

        return logits

if __name__ == "__main__":
    
    # sample_LSTM = LSTM(3, 10, 100, 1000, 300, DEVICE)
    # print(sample_LSTM)
    
    sample_biLSTM = biLSTM(10, 100, 1000, 300, DEVICE)
    print(sample_biLSTM)
    
    TEXT, vocab_size, train_iter, valid_iter, test_iter = load_dataset(seed=SEED)
    # Hyperparameters
    learning_rate = 0.0003
    layers = 2
    output_size = 2
    hidden_size = 16
    embedding_length = 100
    epochs = 10

    # Model, training, testing
    set_seed(SEED)
    '''
    # Train the lSTM model
    lstm_model = LSTM(layers, output_size, hidden_size, vocab_size,
                      embedding_length, DEVICE)
    '''
    # Train the biLSTM model
    lstm_model = biLSTM(output_size, hidden_size, vocab_size, 
                        embedding_length, DEVICE)

    lstm_model.to(DEVICE)
    lstm_train_loss, lstm_train_acc, lstm_validation_loss, lstm_validation_acc = train(lstm_model,
                                                                                       DEVICE,
                                                                                       train_iter,
                                                                                       valid_iter,
                                                                                       epochs,
                                                                                       learning_rate)
    test_accuracy = test(lstm_model, DEVICE, test_iter)
    print(f'\n\nTest Accuracy: {test_accuracy} of the LSTM model\n')
