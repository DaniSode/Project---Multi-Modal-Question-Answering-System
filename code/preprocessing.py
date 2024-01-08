#!/usr/bin/env python
# coding: utf-8

# ## Reduce the size of the dataset and preprocess the data
# 

# Import

# In[1]:


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

# Excel
import pandas as pd
import csv
import ast
import openpyxl


# Define paths

# In[2]:


img_pth = 'dataset/img'
ann_pth = 'dataset/ann'
qst_pth = 'dataset/qst'
out_img_pth = 'preprocessed/img'
out_data_pth = 'preprocessed/data'
out_vocab_pth = 'preprocessed/vocab'
out_ann_pth = 'preprocessed/ann'
out_qst_pth = 'preprocessed/qst'
ckpt_pth = 'late_fusion/ckpt'



def preprocessing(image_dir, annotation_dir, question_dir, output_dir, vocab_dir):
    
    dataset = dict()
    for file in os.scandir(question_dir):
        info = dict()
        
        if 'test' in file.name:
            continue
        elif 'train' in file.name:
            datatype = 'train'
        elif 'val' in file.name:
            datatype = 'val'
        
        with open(file.path, 'r') as f:
            data = json.load(f)
            questions = data['questions'].values()
    
        for ann in os.scandir(annotation_dir):
            if datatype == 'train' and 'train' in ann.name:
                with open(ann.path) as f:
                    annotations = json.load(f)['annotations'].values()
            elif datatype == 'val' and 'val' in ann.name:
                with open(ann.path) as f:
                    annotations = json.load(f)['annotations'].values()
        question_dict = {ans['question_id']: ans for ans in annotations}
        
        match_top_ans.unk_ans = 0
        num = 0
        for idx, qu in enumerate(questions):
            if (idx+1) % 1500 == 0:
                print(f'Processing {datatype} data: {idx+1}/{len(questions)}')
            qu_id = qu['question_id']
            qu_sentence = qu['question']
            qu_tokens = tokenizer(qu_sentence)
            img_id = qu['image_id']
            for dir in os.scandir(image_dir):
                if 'train' == datatype and 'train' in dir.name:
                    dir_path = dir.path
                elif 'val' == datatype and 'val' in dir.name:
                    dir_path = dir.path
                else:
                    continue
                for img in os.scandir(dir_path):
                    components = img.name.split('_')
                    image_id = components[-1].split('.')[0]
                    numeric_part = int(image_id)
                    if img_id == numeric_part:
                        img_path = img.path
            annotation_ans = question_dict[qu_id]['answers']
            
            qu_info = dict()
            qu_info.update({'img_id': img_id,
                            'img_path': img_path,
                            'qu_id': qu_id,
                            'qu_sentence': qu_sentence,
                            'qu_tokens': qu_tokens})
            
            for voc in os.scandir(vocab_dir):
                if 'ann' in voc.name:
                    if datatype == 'train' and 'train' in voc.name:
                        voc_path = voc.path
                    elif datatype == 'val' and 'val' in voc.name:
                        voc_path = voc.path
            
            all_ans, valid_ans = match_top_ans(annotation_ans, voc_path)
            qu_info.update({'all_ans': list(all_ans),
                            'valid_ans': list(valid_ans)})   

            info.update({idx: qu_info})
            
        dataset.update({datatype: info})
        print(f'Total {match_top_ans.unk_ans} out of {len(questions)} answers are <unk>')

    np.save(output_dir + '/train.npy', np.array(dataset['train']))
    np.save(output_dir + '/val.npy', np.array(dataset['val']))
    with open(output_dir + '/train.json', 'w') as f:
        json.dump(dataset['train'], f)
    with open(output_dir + '/val.json', 'w') as f:
        json.dump(dataset['val'], f)

    # Write to excel file
    train_json_path = output_dir + "/train.json"
    val_json_path = output_dir + "/val.json"
    excelwriting(train_json_path, output_dir) 
    excelwriting(val_json_path, output_dir)  


def tokenizer(sentence):

    regex = re.compile(r'(\W+)')
    tokens = regex.split(sentence.lower())
    tokens = [w.strip() for w in tokens if len(w.strip()) > 0]
    return tokens

def match_top_ans(annotation_ans, vocab_path):
    
    if "top_ans" not in match_top_ans.__dict__:
        with open(vocab_path, 'r') as f:
            match_top_ans.top_ans = {line.strip() for line in f}
    annotation_ans = {ans['answer'] for ans in annotation_ans}
    valid_ans = match_top_ans.top_ans & annotation_ans

    if len(valid_ans) == 0:
        valid_ans = ['<unk>']
        match_top_ans.unk_ans += 1

    return annotation_ans, valid_ans

def excelwriting(json_path, output_dir):

    with open(json_path) as f:
        json_data = json.load(f)

    values = json_data.values()
    keys = json_data.keys()

    #Sorting things out
    Index=json_data.keys()
    indexs=[]
    for i in Index:
        indexs.append(int(i))
    
    imgId=[]
    img_path=[]
    quId=[]
    qu_S=[]
    qu_T=[]
    All_A=[]
    valid_A=[]

    for i in Index:
        imgId.append(json_data[i]['img_id'])
        img_path.append(json_data[i]['img_path'])
        quId.append(json_data[i]['qu_id'])
        qu_S.append(json_data[i]['qu_sentence'])
        qu_T.append(json_data[i]['qu_tokens'])
        All_A.append(json_data[i]['all_ans'])
        valid_A.append(json_data[i]['valid_ans'])
    
    #Creating the csv file
    string = json_path.split('.')[0]
    csv_file_path = f'{string}.csv'

    data_list = list(zip(indexs, imgId,img_path,quId,qu_S,qu_T,All_A,valid_A))

    with open(csv_file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write the header
        writer.writerow(["index", "img_id", "img_path","qu_id", "qu_sentence","qu_tokens","all_ans","valid_ans"])

        # Write the data
        writer.writerows(data_list)

def main():

    image_dir = out_img_pth
    annotation_dir = out_ann_pth
    question_dir = out_qst_pth
    output_dir = out_data_pth
    vocab_dir = out_vocab_pth
    preprocessing(image_dir, annotation_dir, question_dir, output_dir, vocab_dir)

main()

