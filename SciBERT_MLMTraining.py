"""
Script to further fine-tune the masked language model of the SciBERT model.
Saves the output.
"""
import os
import wget
import torch
import numpy as np
import pandas as pd
import argparse
import glob
import logging
import pickle
import re

from time import time
from torch import nn
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, Dataset


from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizer)

bert_layer = BertForMaskedLM.from_pretrained('allenai/scibert_scivocab_uncased', output_hidden_states=True)
bert_config = BertConfig()

class LoadDataSet(Dataset):

    def __init__(self, filename, maxlen=64):
        self.df = pd.read_csv(filename, encoding='utf-8')
        self.tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        self.maxlen = maxlen

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        sentence = self.df.loc[index, 'Processed Text']

        tokens = self.tokenizer.tokenize(sentence)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))]
        else:
            tokens = tokens[:self.maxlen-1] + ['[SEP]']

        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        token_ids_tensor = torch.tensor(tokens_ids)

        attn_mask = (token_ids_tensor != 0).long()

        return token_ids_tensor, attn_mask

train_set = LoadDataSet(filename='Test_Text1.csv', maxlen=64)
train_loader = DataLoader(train_set, batch_size=8)

class MaskedLM(nn.Module):
    def __init__(self):
        super(MaskedLM, self).__init__()

        self.bert_layer = BertForMaskedLM.from_pretrained('allenai/scibert_scivocab_uncased', output_hidden_states=True)

        self.dropout = nn.Dropout(0.1)

    def forward(self, seq, attn_masks):

        loss, logits, output = self.bert_layer(seq, attention_mask=attn_masks, masked_lm_labels=seq)

        return loss, logits, output

model = MaskedLM()

criterion = BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=0.00002)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

torch.cuda.empty_cache()

def train(model, optimizer, train_loader, device, epochs=3, print_every=100):

    model.to(device)

    model.train()

    print('Training is starting')

    for epoch in range(epochs):

        print('Epoch {}'.format(epoch))
        total_train_loss = 0

        for i, (seq, attn_masks) in enumerate(train_loader):

            optimizer.zero_grad()

            seq, attn_masks = seq.to(device), attn_masks.to(device)
            # print(seq.size()), print(attn_masks)
            loss, logits, output = model(seq, attn_masks)

            total_train_loss += loss.item()

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()

            if (i + 1) % print_every == 0:
                print("Iteration {} ==== Loss {}".format(i + 1, loss.item()))

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_loader)
        print("  Average training loss: {0:.6f}".format(avg_train_loss))


train(model, optimizer, train_loader, device, epochs=1, print_every=100)

if not os.path.isdir('D:/Dan/PythonProjects/SciBERT_CORD19/Models/'):
    os.mkdir('D:/Dan/PythonProjects/SciBERT_CORD19/Models/')

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, 'model.pth')

# Loading the checkpoints for resuming training
# checkpoint = torch.load('model.pth')
# model.load_state_dict(checkpoint['model_state_dict'])
