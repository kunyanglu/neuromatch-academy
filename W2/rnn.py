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

SEED = 2021
DEVICE = 'cpu'

class VanillaRNN(nn.Module):
    def __init__(self, layers, output_size, hidden_size, vocab_size, 
                        embed_size, device):

        super(VanillaRNN, self).__init__()
        self.n_layers= layers
        self.hidden_size = hidden_size
        self.device = device

        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, self.n_layers)
        self.fc = nn.Linear(self.n_layers*hidden_size, output_size)

    def forward(self, inputs):
        input = self.embeddings(inputs)
        input = input.permute(1, 0, 2)
        h_0 = torch.zeros(2, input.size()[1], self.hidden_size).to(self.device)
        output, h_n = self.rnn(input, h_0)
        h_n = h_n.permute(1, 0, 2)
        h_n = h_n.contiguous().reshape(h_n.size()[0], h_n.size()[1]*h_n.size()[2])
        logits = self.fc(h_n)

        return logits

def load_dataset(sentence_length=50, batch_size=32, seed=522):

    TEXT = data.Field(sequential=True,
                      tokenize=nltk.word_tokenize,
                      lower=True,
                      include_lengths=True,
                      batch_first=True,
                      fix_length=sentence_length)
    LABEL = data.LabelField(dtype=torch.float)

    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

    TEXT.build_vocab(train_data)
    LABEL.build_vocab(train_data)

    train_data, valid_data = train_data.split(split_ratio=0.7,
                                    random_state=random.seed(seed))

    train_iter, valid_iter, test_iter = data.BucketIterator.splits(
                                    (train_data, valid_data, test_data),
                                    batch_size=batch_size,
                                    sort_key=lambda x: len(x.text),
                                    repeat=False,
                                    shuffle=True)
    vocab_size = len(TEXT.vocab)

    print(f"Data loading is completed. Sentence length: {sentence_length}, "
            f"Batch size: {batch_size}, and seed: {seed}")

    return TEXT, vocab_size, train_iter, valid_iter, test_iter

def train(model, device, train_iter, valid_iter, epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss, validation_loss = [], []
    train_acc, validation_acc = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.
        correct, total = 0, 0
        steps = 0

        for idx, batch in enumerate(train_iter):
            text = batch.text[0]
            target = batch.label
            target = torch.autograd.Variable(target).long()
            text, target = text.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(text)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            steps += 1
            running_loss += loss.item()

            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        train_loss.append(running_loss/len(train_iter))
        train_acc.append(correct/total)

        print(f'Epoch: {epoch + 1}, '
              f'Training Loss: {running_loss/len(train_iter):.4f}, '
              f'Training Accuracy: {100*correct/total: .2f}%')

        # evaluate on validation data
        model.eval()
        running_loss = 0.
        correct, total = 0, 0

        with torch.no_grad():
            for idx, batch in enumerate(valid_iter):
                text = batch.text[0]
                target = batch.label
                target = torch.autograd.Variable(target).long()
                text, target = text.to(device), target.to(device)

                optimizer.zero_grad()
                output = model(text)

                loss = criterion(output, target)
                running_loss += loss.item()

                # get accuracy
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        validation_loss.append(running_loss/len(valid_iter))
        validation_acc.append(correct/total)

        print (f'Validation Loss: {running_loss/len(valid_iter):.4f}, '
               f'Validation Accuracy: {100*correct/total: .2f}%')

    return train_loss, train_acc, validation_loss, validation_acc

def test(model, device, test_iter):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for idx, batch in enumerate(test_iter):
            text = batch.text[0]
            target = batch.label
            target = torch.autograd.Variable(target).long()
            text, target = text.to(device), target.to(device)

            outputs = model(text)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        acc = 100 * correct / total
        return acc

if __name__ == "__main__":

    TEXT, vocab_size, train_iter, valid_iter, test_iter = load_dataset(seed=SEED)
    sampleRNN = VanillaRNN(2, 10, 50, 1000, 300, DEVICE)
    print(sampleRNN)
    
    learning_rate = 0.0002
    layers = 2
    output_size = 2
    hidden_size = 50  # 100
    embedding_length = 100
    epochs = 10
    
    # Initialize model, training and testing
    set_seed(SEED)
    vanilla_rnn_model = VanillaRNN(layers, output_size, hidden_size, vocab_size,
                                   embedding_length, DEVICE)
    vanilla_rnn_model.to(DEVICE)
    vanilla_rnn_start_time = time.time()
    vanilla_train_loss, vanilla_train_acc, vanilla_validation_loss, vanilla_validation_acc = train(vanilla_rnn_model,
                                                                                                   DEVICE,
                                                                                                   train_iter,
                                                                                                   valid_iter,
                                                                                                   epochs,
                                                                                                   learning_rate)

    print("--- Time taken to train = %s seconds ---" % (time.time() - vanilla_rnn_start_time))
    test_accuracy = test(vanilla_rnn_model, DEVICE, test_iter)
    print(f'Test Accuracy: {test_accuracy} with len=50\n')

    # Number of model parameters
    print(f'Number of parameters = {count_parameters(vanilla_rnn_model)}')

