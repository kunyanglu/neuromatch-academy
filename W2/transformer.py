import tqdm
import math
import torch
import statistics

import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch import nn
from pprint import pprint
from tqdm.notebook import tqdm
from datasets import load_metric
from datasets import load_dataset

# transformers library
from transformers import Trainer
from transformers import pipeline
from transformers import set_seed
from transformers import AutoTokenizer
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM
from transformers import AutoModelForSequenceClassification

# pytorch
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert import BertForMaskedLM

# textattack

class Transformer(nn.Module):
    
    def __init__(self, k, heads, depth, seq_length, num_tokens, num_classes):
        super().__init__()
        
        self.k = k
        self.num_tokens = num_tokens
        self.token_embedding = nn.Embedding(num_tokens, k)
        self.pos_enc = PositionalEncoding(k)

        transformer_blocks = []
        for i in range(depth):
            transformer_blocks.append(TransformerBlock(k=k, heads=heads))

        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        self.classification_head = nn.Linear(k, num_classes)

    def forward(self, x):
        x = self.token_embedding(x) * np.sqrt(self.k)
        x = self.pos_enc(x)
        x = self.transformer_blocks(x)
        x = x.mean(dim=1)
        x = self.classification_head(sequence_avg)
        logprobs = F.log_softmax(x, dim=1)

        return logprobs

class DotProductAttention(nn.Module):
    
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, b, h, t, k):
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
        values = values.transpose(1, 2).contiguous().view(b * h, t, k)

        score = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(k)
        softmax_weights = F.softmax(score, dim=2)
        
        out = torch.bmm(self.dropout(softmax_weights), values).view(b, h, t, k)
        out = out.transpose(1, 2).contiguous().view(b, t, h * k)

        return out

class TransformerBlock(nn.Module):

    '''Transformer Block'''
    def __init__(self, k, heads):
        super().__init__()

        self.attention = SelfAttention(k, heads=heads)

        self.norm_1 = nn.LayerNorm(k)
        self.norm_2 = nn.LayerNorm(k)
        
        # arbitrarily choose a hidden size
        hidden_size = 2 * k
        self.mlp = nn.Sequential(
                nn.Linear(k, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, k))

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm_1(attended + x)
        feedforward = self.mlp(x)
        x = self.norm_2(feedforward + x)

        return x
    
class SelfAttention(nn.Module):
    '''Muti-head self attention layer '''
    def __init__(self, k, heads=8, dropout=0.1):
        super().__init__()
        self.k, self.heads = k, heads

        self.to_keys = nn.Linear(k, k * heads, bias=False)
        self.to_queries = nn.Linear(k, k * heads, bias=False)
        self.to_values = nn.Linear(k, k * heads, bias=False)
        self.unify_heads = nn.Linear(k * heads, k)

        self.attention = DotProductAttention(dropout)

    def forward(self, x):
        b, t, k = x.size()
        h = self.heads

        queries = self.to_queries(x).view(b, t, h, k)
        keys = self.to_keys(x).view(b, t, h, k)
        values = self.to_values(x).view(b, t, h, k)

        out = self.attention(queries, keys, values, b, h, t, k)

        return self.unify_heads(out)

class PositionalEncoding(nn.Module):
    
    def __init__(self, emb_size, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, emb_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-np.log(10000.0) / emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def train(model, loss_fn, train_loader, 
          n_iter=1, learning_rate=1e-4,
          test_loader=None, device='cpu',
          L2_penalty=0, L1_penalty=0):

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss = []
    test_loss = []

    for iter in range(n_iter):
        iter_train_loss = []
        for i, batch in tqdm(enumerate(train_loader)):
            out = model(batch['input_ids'].to(device))
            loss = loss_fn(out, batch['label'].to(device))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_train_loss.append(loss.item())
    
        ### Test ###
        if test_loader is not None:
            print('Running Test loop')
            iter_loss_test = []
            for j, test_batch in enumerate(test_loader):

                out_test = model(test_batch['input_ids'].to(device))
                loss_test = loss_fn(out_test, test_batch['label'].to(device))
                iter_loss_test.append(loss_test.item())

            test_loss.append(statistics.mean(iter_loss_test))
            
            if test_loader is None:
                print(f'iteration {iter + 1}/{n_iter} | train loss: {loss.item():.3f}')
            else:
                print(f'iteration {iter + 1}/{n_iter} | train loss: {loss.item():.3f} | test_loss: {loss_test.item():.3f}')

    if test_loader is None:
        return train_loss
    else:
        return train_loss, test_loss

if __name__ == "__main__":
    
    max_len = 32
    vocab_size = 28996
    num_classes = 32
    # Set random seeds for reproducibility
    np.random.seed(1)
    torch.manual_seed(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize network with embedding size 128, 8 attention heads, and 3 layers
    model = Transformer(128, 8, 3, max_len, vocab_size, num_classes).to(device)

    # Initialize built-in PyTorch Negative Log Likelihood loss function
    loss_fn = F.nll_loss
    ''' Unable to load data 
    train_loss, test_loss = train(model, loss_fn, train_loader, test_loader=test_loader,
                                  device=device)'''
