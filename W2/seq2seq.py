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

class Seq2SeqEncoder(d2l.Encoder):
     
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                    dropout=0, **kwargs):
        
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, 
                            dropout=dropout)

    def forward(self, X, *args):
        # The output `X` shape: (`batch_size`, `num_steps`, `embed_size`)
        X = self.embedding(X)
        # In RNN models, the first axis corresponds to time steps
        X = X.permute(1, 0, 2)

        output, state = self.rnn(X)

        return output, state

class Seq2SeqDecoder(d2l.Decoder):

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                     dropout=0, **kwargs):

        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                            dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]
    
    def forward(self, X, state):
        # The output `X` shape: (`num_steps`, `batch_size`, `embed_size`)
        X = self.embedding(X).permute(1, 0, 2)
        # Broadcast `context` so it has the same `num_steps` as `X`
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)

        return output, state

###################################
### Section 3.2 - Loss Function ###
###################################

### Mask irrelevant entries of the padding tokens with zeros
### thus exclude in loss computation

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):

    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)

        return weighted_loss

def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

if __name__ == "__main__":
    
    X = torch.zeros((4, 7), dtype=torch.long)
    encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
    encoder.eval()
    output, state = encoder(X)
    print(output.shape)
    print(state.shape)
        
    decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                             num_layers=2)
    # decoder.initialize()
    state = decoder.init_state(encoder(X))
    output, state = decoder(X, state)
    print(f"Output's shape: {output.shape}, Number of states: {len(state)}, State's shape: {state[0].shape}")

    ########################
    ### Test Section 3.2 ###
    ########################
    
    X = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(sequence_mask(X, torch.tensor([1, 3])))
    X = torch.ones(2, 3, 4)
    print(sequence_mask(X, torch.tensor([1, 2]), value=-1))

    loss = MaskedSoftmaxCELoss()
    print(loss(torch.ones(3, 4, 10),
         torch.ones((3, 4), dtype=torch.long),
         torch.tensor([4, 2, 0])))

