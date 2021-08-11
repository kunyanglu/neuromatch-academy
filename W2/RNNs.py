# Imports
import time
import math
import torch
import string
import random
import unidecode
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm.notebook import tqdm
from helper import read_file, char_tensor, time_since, generate

SEED = 2021

class CharRNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size,
                model="gru", n_layers=1):

        super(CharRNN, self).__init__()
        self.model = model.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        
        if self.model == "gru":
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers)
        elif self.model == "lstm":
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers)
        elif self.model == "rnn":
            self.rnn = nn.RNN(hidden_size, hidden_size, n_layers)

        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        batch_size = input.size(0)
        encoded = self.encoder(input)
        output, hidden = self.rnn(encoded.reshape(1, batch_size, -1), hidden)
        output = self.decoder(output.reshape(batch_size, -1))
        return output, hidden

    def init_hidden(self, batch_size):
        if self.model == "lstm":
            return (torch.zeros(self.n_layers, batch_size, self.hidden_size), \
                    torch.zeros(self.n_layers, batch_size, self.hidden_size))

        return torch.zeros(self.n_layers, batch_size, self.hidden_size)

def random_training_set(file, file_len, chunk_len, batch_size,
                        device='cpu', seed=0):
    random.seed(seed)

    inp = torch.LongTensor(batch_size, chunk_len).to(device)
    target = torch.LongTensor(batch_size, chunk_len).to(device)

    for bi in range(batch_size):
        start_index = random.randint(0, file_len - chunk_len - 1)
        end_index = start_index + chunk_len + 1
        chunk = file[start_index:end_index]
        inp[bi] = char_tensor(chunk[:-1])
        target[bi] = char_tensor(chunk[1:])

    return inp, target, chunk_len, batch_size, device

def train(decoder, criterion, inp, target, chunk_len, batch_size, device):
    hidden = decoder.init_hidden(batch_size)
    decoder.zero_grad()
    loss = 0

    for c in range(chunk_len):
        output, hidden = decoder(inp[:, c].to(device), hidden.to(device))
        loss += criterion(output.reshape(batch_size, -1), target[:,c])

    loss.backward()
    decoder_optimizer.step()
    return loss.item() / chunk_len

if __name__ == "__main__":
    # Reading and un-unicode-encoding data
    all_characters = string.printable
    n_characters = len(all_characters)

    # Load the text file
    file, file_len = read_file('twain.txt')

    # Hyperparams
    batch_size = 50
    chunk_len = 200
    model = "rnn"  # other options: `lstm`, `gru`

    n_layers = 2
    hidden_size = 200
    learning_rate = 0.01

    # Define the model, optimizer, and the loss criterion
    decoder = CharRNN(n_characters, hidden_size, n_characters,
                      model=model, n_layers=n_layers)
    decoder.to('cpu')

    decoder_optimizer = torch.optim.Adagrad(decoder.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    n_epochs = 1000   # initial was set to 2000

    print_every = 50  # frequency of printing the outputs

    start = time.time()
    all_losses = []
    loss_avg = 0

    print(f"Training for {n_epochs} epochs...\n")
    
    for epoch in tqdm(range(1, n_epochs + 1), position=0, leave=True):
        loss = train(decoder, criterion,
               *random_training_set(file, file_len, chunk_len, batch_size,
                                    device='cpu', seed=epoch))
        loss_avg += loss

    if epoch % print_every == 0:
        print(f"[{time_since(start)} {epoch/n_epochs * 100}%) {loss:.4f}]")
        print(f"{generate(decoder, prime_str='Wh', predict_len=150, device='cpu')}")


