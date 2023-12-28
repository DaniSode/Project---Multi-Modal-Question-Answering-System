#!/usr/bin/env python
# coding: utf-8

# ## Reduce the size of the dataset and preprocess the data
# 

# Import

# In[ ]:


# Resizing img
import os
import argparse
from PIL import Image
from sklearn.model_selection import train_test_split

# Make vocab
import json
import re
from collections import defaultdict

# Preprocess data
import glob
import numpy as np

# Build dataset
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

# Models
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# Training
import time
from torch import optim
import tensorflow as tf
from tensorflow.keras.applications import VGG19


# Define paths

# In[ ]:


img_pth = 'dataset/img'
ann_pth = 'dataset/ann'
qst_pth = 'dataset/qst'
out_img_pth = 'preprocessed/img'
out_data_pth = 'preprocessed/data'
out_vocab_pth = 'preprocessed/vocab'
out_ann_pth = 'preprocessed/ann'
out_qst_pth = 'preprocessed/qst'
ckpt_pth = 'late_fusion/ckpt'
log_pth = 'late_fusion/log'


# Build Dataset

# In[ ]:


class VQADataset(Dataset):

    def __init__(self, input_dir, input_file, max_qu_len = 30, transform = None):

        with open(os.path.join(input_dir, input_file), 'r') as file:
            self.input_data = json.load(file)
        self.qu_vocab = Vocab('preprocessed/vocab/qst_vocabs.txt')
        self.ans_vocab = Vocab('preprocessed/vocab/ann_vocabs.txt')
        self.max_qu_len = max_qu_len
        self.transform = transform
        self.labeled = True if not "test" in input_file else False  #added this

    def __getitem__(self, idx):
        idx = str(idx)
        path = self.input_data[idx]['img_path']
        img = np.array(Image.open(path).convert('RGB'))
        qu_id = self.input_data[idx]['qu_id']
        qu_tokens = self.input_data[idx]['qu_tokens']
        qu2idx = np.array([self.qu_vocab.word2idx('<pad>')] * self.max_qu_len)
        qu2idx[:len(qu_tokens)] = [self.qu_vocab.word2idx(token) for token in qu_tokens]
        sample = {'image': img, 'question': qu2idx, 'question_id': qu_id}

        if self.labeled:
            print('ans2idx')
            ans2idx = [self.ans_vocab.word2idx(ans) for ans in self.input_data[idx]['valid_ans']]
            print(np.shape(ans2idx))
            #ans2idx = np.random.choice(ans2idx)
            ans2idx = np.random.choice(ans2idx, size=1)
            print(np.shape(ans2idx))
            sample['answer'] = ans2idx
        #old
        #ans2idx = [self.ans_vocab.word2idx(ans) for ans in self.input_data[idx]['valid_ans']]
        #ans2idx = np.random.choice(ans2idx, size=1, replace=False)
        #ans2idx = np.random.choice(ans2idx)
        #sample['answer'] = ans2idx

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

    def __len__(self):

        return len(self.input_data)


def data_loader(input_dir, batch_size, max_qu_len, num_worker):

    transform = transforms.Compose([
        transforms.ToTensor(),  # convert to (C,H,W) and [0,1]
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # mean=0; std=1
    ])

    vqa_dataset = {
        'train': VQADataset(
            input_dir=input_dir,
            input_file='train.json',
            max_qu_len=max_qu_len,
            transform=transform),
        'val': VQADataset(
            input_dir=input_dir,
            input_file='val.json',
            max_qu_len=max_qu_len,
            transform=transform)
    }

    dataloader = {
        key: DataLoader(
            dataset=vqa_dataset[key],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_worker)
        for key in ['train', 'val']
    }

    return dataloader

class Vocab:

    def __init__(self, vocab_file):

        self.vocab = self.load_vocab(vocab_file)
        self.vocab2idx = {vocab: idx for idx, vocab in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

    def load_vocab(self, vocab_file):

        with open(vocab_file) as f:
            vocab = [v.strip() for v in f]

        return vocab

    def word2idx(self, vocab):

        if vocab in self.vocab2idx:
            return self.vocab2idx[vocab]
        else:
            return ['<unk>']

    def idx2word(self, idx):

        return self.vocab[idx]


# Late fusion models

# In[ ]:


class ImgEncoder(nn.Module):

    def __init__(self, embed_dim):

        super(ImgEncoder, self).__init__()
        self.model = models.vgg19(weights='VGG19_Weights.DEFAULT')
        in_features = self.model.classifier[-1].in_features
        self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-1]) # remove vgg19 last layer
        self.fc = nn.Linear(in_features, embed_dim)

    def forward(self, image):

        with torch.no_grad():
            img_feature = self.model(image) # (batch, channel, height, width)
        img_feature = self.fc(img_feature)

        l2_norm = F.normalize(img_feature, p=2, dim=1).detach()
        return l2_norm

class QuEncoder(nn.Module):

    def __init__(self, qu_vocab_size, word_embed, hidden_size, num_hidden, qu_feature_size):

        super(QuEncoder, self).__init__()
        self.word_embedding = nn.Embedding(qu_vocab_size, word_embed)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed, hidden_size, num_hidden) # input_feature, hidden_feature, num_layer
        self.fc = nn.Linear(2*num_hidden*hidden_size, qu_feature_size)

    def forward(self, question):

        qu_embedding = self.word_embedding(question)                # (batchsize, qu_length=30, word_embed=300)
        qu_embedding = self.tanh(qu_embedding)
        qu_embedding = qu_embedding.transpose(0, 1)                 # (qu_length=30, batchsize, word_embed=300)
        _, (hidden, cell) = self.lstm(qu_embedding)                 # (num_layer=2, batchsize, hidden_size=1024)
        qu_feature = torch.cat((hidden, cell), dim=2)               # (num_layer=2, batchsize, 2*hidden_size=1024)
        qu_feature = qu_feature.transpose(0, 1)                     # (batchsize, num_layer=2, 2*hidden_size=1024)
        qu_feature = qu_feature.reshape(qu_feature.size()[0], -1)   # (batchsize, 2*num_layer*hidden_size=2048)
        qu_feature = self.tanh(qu_feature)
        qu_feature = self.fc(qu_feature)                            # (batchsize, qu_feature_size=1024)

        return qu_feature

class VQAModel(nn.Module):

    def __init__(self, feature_size, qu_vocab_size, ans_vocab_size, word_embed, hidden_size, num_hidden):

        super(VQAModel, self).__init__()
        self.img_encoder = ImgEncoder(feature_size)
        self.qu_encoder = QuEncoder(qu_vocab_size, word_embed, hidden_size, num_hidden, feature_size)
        self.dropout = nn.Dropout(0.5)
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(feature_size, ans_vocab_size)
        self.fc2 = nn.Linear(ans_vocab_size, ans_vocab_size)

    def forward(self, image, question):

        img_feature = self.img_encoder(image)               # (batchsize, feature_size=1024)
        qst_feature = self.qu_encoder(question)
        combined_feature = img_feature * qst_feature
        combined_feature = self.dropout(combined_feature)
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.fc1(combined_feature)       # (batchsize, ans_vocab_size=1000)
        combined_feature = self.dropout(combined_feature)
        combined_feature = self.tanh(combined_feature)
        logits = self.fc2(combined_feature)                 # (batchsize, ans_vocab_size=1000)

        return logits


# Training loop

# In[ ]:


#BATCH_SIZE = 70
BATCH_SIZE = 150
MAX_QU_LEN = 30
NUM_WORKER = 8
#FEATURE_SIZE, WORD_EMBED = 512, 150
FEATURE_SIZE, WORD_EMBED = 1024, 300
#NUM_HIDDEN, HIDDEN_SIZE = 2, 128
NUM_HIDDEN, HIDDEN_SIZE = 2, 512
LEARNING_RATE, STEP_SIZE, GAMMA = 0.001, 10, 0.1
EPOCH = 50

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train():

    dataloader = data_loader(input_dir=out_data_pth, batch_size=BATCH_SIZE, max_qu_len=MAX_QU_LEN, num_worker=NUM_WORKER)
    qu_vocab_size = dataloader['train'].dataset.qu_vocab.vocab_size
    ans_vocab_size = dataloader["train"].dataset.ans_vocab.vocab_size

    model = VQAModel(feature_size=FEATURE_SIZE, qu_vocab_size=qu_vocab_size, ans_vocab_size=ans_vocab_size,
                     word_embed=WORD_EMBED, hidden_size=HIDDEN_SIZE, num_hidden=NUM_HIDDEN).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    criterion = nn.CrossEntropyLoss()

    print('>> start training')
    start_time = time.time()
    for epoch in range(EPOCH):
        epoch_loss = {key: 0 for key in ['train', 'val']}
        model.train()
        for idx, sample in enumerate(dataloader['train']):
            image = sample['image'].to(device=device)
            question = sample['question'].to(device=device)
            #label = sample['answer'].to(device=device)
            label = sample['answer'].squeeze().view(-1).to(device=device)
            
            print('hey')
            print(np.shape(image))
            print(np.shape(question))
            print(np.shape(label))
            # forward
            logits = model(image, question)
            print(np.shape(logits))
            loss = criterion(logits, label)
            epoch_loss['train'] += loss.item()
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        for idx, sample in enumerate(dataloader['val']):

            image = sample['image'].to(device=device)
            question = sample['question'].to(device=device)
            label = sample['answer'].to(device=device)
            with torch.no_grad():
                logits = model(image, question)
                loss = criterion(logits, label)
            epoch_loss['val'] += loss.item()
        # statistic
        for phase in ['train', 'val']:
            epoch_loss[phase] /= len(dataloader[phase])
            with open(os.path.join(log_pth, f'{phase}_log.txt'), 'a') as f:
                f.write(str(epoch+1) + '\t' + str(epoch_loss[phase]) + '\n')
        print('Epoch:{}/{} | Training Loss: {train:6f} | Validation Loss: {val:6f}'.format(epoch+1, EPOCH, **epoch_loss))

        scheduler.step()
        early_stop = early_stopping(model, epoch_loss['val'])
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(ckpt_pth, f'model-epoch-{epoch+1}.pth'))
        if early_stop:
            print(f'>> Early stop at {epoch+1} epoch')
            break

    end_time = time.time()
    training_time = end_time - start_time
    print(f">> Finishing training | Training Time:{training_time//60:.0f}m:{training_time%60:.0f}s")

def early_stopping(model, epoch_loss, patience=7):

    early_stop = False
    if not bool(early_stopping.__dict__):
        early_stopping.best_loss = epoch_loss
        early_stopping.record_loss = epoch_loss
        early_stopping.counter = 0

    if epoch_loss < early_stopping.best_loss:
        early_stopping.best = epoch_loss
        torch.save(model.state_dict(), os.path.join(ckpt_pth, 'best_model.pth'))

    if epoch_loss > early_stopping.record_loss:
        early_stopping.counter += 1
        if early_stopping.counter >= patience:
            early_stop = True
    else:
        early_stopping.counter = 0
        early_stopping.record_loss = epoch_loss

    return early_stop

if __name__ == '__main__':

    if not os.path.exists(log_pth):
        os.makedirs(log_pth)
    if not os.path.exists(ckpt_pth):
        os.makedirs(ckpt_pth)
    train()

