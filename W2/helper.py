import numpy as np
import random
import requests
import torch
import unidecode
import time
import math
import string

import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm.notebook import tqdm
from nltk.tokenize import word_tokenize
from torchtext import data, datasets

def set_seed(seed=None, seed_torch=True):

    if seed is None:
        seed = np.random.choice(2 ** 32)
        random.seed(seed)
        np.random.seed(seed)
    
    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    print(f'Random seed {seed} has been set.')

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

#################### 
##### W2D1 RNN #####
####################

def read_file(filename):
    file = unidecode.unidecode(open(filename).read())
    return file, len(file)

# Turning a string into a tensor
def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = all_characters.index(string[c])
        except:
            continue
        
    return tensor

# Readable time elapsed
def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    out = f"{m}min {s}sec"
    return out

def generate(decoder, prime_str='A', predict_len=100, temperature=0.8,
             device='cpu'):

    hidden = decoder.init_hidden(1)
    prime_input = char_tensor(prime_str).unsqueeze(0)

    hidden = hidden.to(device)
    prime_input = prime_input.to(device)
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[:,p], hidden)

    inp = prime_input[:,-1]

    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = char_tensor(predicted_char).unsqueeze(0)
        inp = inp.to(device)

    return predicted

#################################################
### W2D3 Modeling Sequences and Encoding Text ###
#################################################

def cosine_similarity(vec_a, vec_b):
    """Compute cosine similarity between vec_a and vec_b"""
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

def tokenize(sentences):
    # Tokenize the sentence
    # from nltk.tokenize library use word_tokenize
    token = word_tokenize(sentences)

    return token

def plot_train_val(x, train, val, train_label, val_label, title, y_label,
                   color):
    plt.plot(x, train, label=train_label, color=color)
    plt.plot(x, val, label=val_label, color=color, linestyle='--')
    plt.legend(loc='lower right')
    plt.xlabel('epoch')
    plt.ylabel(y_label)
    plt.title(title)

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={ 'id': id }, stream=True)
    token = get_confirm_token(response)

    if token:
        params = { 'id': id, 'confirm': token }
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks\
                    f.write(chunk)

if __name__ == "__main__":

    pass

