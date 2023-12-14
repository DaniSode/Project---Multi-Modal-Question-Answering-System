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


# Define paths

# In[2]:


img_pth = 'dataset\img'
ann_pth = 'dataset\\ann'
qst_pth = 'dataset\qst'
out_img_pth = 'preprocessed\img'
out_data_pth = 'preprocessed\data'
out_vocab_pth = 'preprocessed\\vocab'
out_ann_pth = 'preprocessed\\ann'
out_qst_pth = 'preprocessed\qst'
ckpt_pth = 'late_fusion\ckpt'
log_pth = 'late_fusion\log'


# Preprocess images

# In[ ]:


def resize_image(image, size):
    """Resize an image to the given size."""
    return image.resize(size, Image.Resampling.LANCZOS)

def resize_images(input_dir, output_dir, size, split_ratio):
    """Resize the images in 'input_dir' and save into 'output_dir'."""
    for idir in os.scandir(input_dir):
        if not idir.is_dir():
            print('No valid directory')
            continue
        if not os.path.exists(output_dir+'\\'+idir.name):
            os.makedirs(output_dir+'\\'+idir.name)
        else:
            for file in os.listdir(output_dir+'\\'+idir.name):
                if os.path.isfile(os.path.join(output_dir+'\\'+idir.name, file)):
                    os.remove(os.path.join(output_dir+'\\'+idir.name, file))
                      
        images = os.listdir(idir.path)
        images, _ = train_test_split(images,
                                     test_size=split_ratio,
                                     shuffle=False)
        n_images = len(images)
        for id, image in enumerate(images):
            try:
                with open(os.path.join(idir.path, image), 'r+b') as f:
                    with Image.open(f) as img:
                        img = resize_image(img, size)
                        img.save(os.path.join(output_dir+'\\'+idir.name, image), img.format)
            except(IOError, SyntaxError) as e:
                pass
            if (id+1) % 500 == 0:
                print("[{}/{}] resized images and saved into '{}'."
                      .format(id+1, n_images, output_dir+'\\'+idir.name))
                    
def main():

    input_dir = img_pth
    output_dir = out_img_pth
    image_size = [224, 224]
    split_ratio = 0.95
    resize_images(input_dir, output_dir, image_size, split_ratio)
    
main()


# Removing questions and annotations that doesnt belong to the images

# In[ ]:


def removing(out_img_pth, ann_pth, qst_pth):

    test_ids = []
    train_ids = []
    val_ids = []
    for idir in os.scandir(out_img_pth):
        for file in os.listdir(idir):
            components = file.split('_')
            image_id = components[-1].split('.')[0]
            numeric_part = int(image_id)
            if 'test' in idir.name:
                test_ids.append(numeric_part)
            elif 'train' in idir.name:
                train_ids.append(numeric_part)
            elif 'val' in idir.name:
                val_ids.append(numeric_part)
    
    for idir in os.scandir(ann_pth):
        if 'test' in idir.name:
            ids = test_ids
        elif 'train' in idir.name:
            ids = train_ids
        elif 'val' in idir.name:
            ids = val_ids
        
        for file in os.listdir(idir):
            path = os.path.join(idir, file)
            with open(path, 'r') as f:
                data = json.load(f)
            annotations = data['annotations']
            prelen = len(annotations)
            labels = dict()
            number = 0
            for label in annotations:
                    if int(label['image_id']) in ids:
                        labels.update({number: label})   
                        number += 1
            print(f'{len(labels)} remaining annotations of {prelen} in {file}')
            data['annotations'] = labels
            with open(out_ann_pth + '\\' + file, 'w') as f:
                json.dump(data, f)
    
    for idir in os.scandir(qst_pth):
        if 'test' in idir.name:
            ids = test_ids
        elif 'train' in idir.name:
            ids = train_ids
        elif 'val' in idir.name:
            ids = val_ids
            
        for file in os.listdir(idir):
            path = os.path.join(idir, file)
            with open(path, 'r') as f:
                data = json.load(f)
            questions = data['questions']
            prelen = len(questions)
            labels = dict()
            number = 0
            for label in questions:
                    if int(label['image_id']) in ids:
                        labels.update({number: label}) 
                        number += 1
            print(f'{len(labels)} remaining questions of {prelen} in {file}')
            data['questions'] = labels
            with open(out_qst_pth + '\\' + file, 'w') as f:
                json.dump(data, f)

def main():
    img_pth = out_img_pth
    annot_pth = ann_pth
    quest_pth = qst_pth
    removing(img_pth, annot_pth, quest_pth)
    
main()


# Make vocab

# In[ ]:


def make_q_vocab(input_dir, output_dir):

    for file in os.scandir(input_dir):
        if "test" in file.name:
            continue
        if not os.path.exists(output_dir):
            os.makedirs(output_dir) 
            
        regex = re.compile(r'(\W+)')
        q_vocab = []
        path = os.path.join(input_dir, file.name)
        with open(path, 'r') as f:
            q_data = json.load(f)
        question = q_data['questions'].values()
        for quest in question:
            split = regex.split(quest['question'].lower())
            tmp = [w.strip() for w in split if len(w.strip()) > 0]
            q_vocab.extend(tmp)
    
        q_vocab = list(set(q_vocab))
        q_vocab.sort()
        q_vocab.insert(0, '<pad>')
        q_vocab.insert(1, '<unk>')
    
        with open(output_dir + '\\' + file.name.split(".")[0] + '_vocabs.txt', 'w') as f:
            f.writelines([v+'\n' for v in q_vocab])

        print(f'The number of total words of questions in {file.name}: {len(q_vocab)}')

def make_a_vocab(input_dir, output_dir):

    for file in os.scandir(input_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir) 
            
        answers = defaultdict(lambda :0)
        path = os.path.join(input_dir, file.name)
        with open(path, 'r') as f:
            data = json.load(f)
        annotations = data['annotations'].values()
        for label in annotations:
            for ans in label['answers']:
                vocab = ans['answer']
                if re.search(r'[^\w\s]', vocab):
                    continue
                answers[vocab] += 1
    
        answers = sorted(answers, key=answers.get, reverse= True) 
        with open(output_dir + '\\' + file.name.split(".")[0] + '_vocabs.txt', 'w') as f :
            f.writelines([ans+'\n' for ans in answers])
            
        print(f'The number of total words of answers in {file.name}: {len(answers)}')

def make_vocab(output_dir):
    ann_vocab = set()
    qst_vocab = set()
    for file in os.scandir(output_dir):
        if 'ann' in file.name:
            with open(file.path, 'r') as f:
                for line in f: 
                    ann_vocab.add(line.split('\n')[0])
        elif 'qst' in file.name:
            with open(file.path, 'r') as f:
                for line in f: 
                    qst_vocab.add(line.split('\n')[0])

    print(f'The number of total words of answers in ann_vocabs will be: {len(ann_vocab)}')
    print(f'The number of total words of answers in qst_vocabs will be: {len(qst_vocab)}')
    
    with open(output_dir + '\\ann_vocabs.txt', 'w') as f:
        f.writelines([ans+'\n' for ans in ann_vocab])
    with open(output_dir + '\\qst_vocabs.txt', 'w') as f:
        f.writelines([ans+'\n' for ans in qst_vocab])

def main():
    input_qst_dir = out_qst_pth
    input_ann_dir = out_ann_pth
    output_vocab_dir = out_vocab_pth
    make_q_vocab(input_qst_dir, output_vocab_dir)
    make_a_vocab(input_ann_dir, output_vocab_dir)
    make_vocab(out_vocab_pth)
main()
    


# Make Vocab

# In[ ]:


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

    np.save(output_dir + '\\train.npy', np.array(dataset['train']))
    np.save(output_dir + '\\val.npy', np.array(dataset['val']))
    with open(output_dir + '\\train.json', 'w') as f:
        json.dump(dataset['train'], f)
    with open(output_dir + '\\val.json', 'w') as f:
        json.dump(dataset['val'], f)

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

def main():

    image_dir = out_img_pth
    annotation_dir = out_ann_pth
    question_dir = out_qst_pth
    output_dir = out_data_pth
    vocab_dir = out_vocab_pth
    preprocessing(image_dir, annotation_dir, question_dir, output_dir, vocab_dir)

main()

