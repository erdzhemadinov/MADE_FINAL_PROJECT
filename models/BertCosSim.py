#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 12:53:55 2020

@author: anton
"""

import pickle
import numpy as np
import torch
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, AutoConfig

class BertCosineSimilraty():

    def __init__(self, bert_model, bert_tokenizer, cpu=False):
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer 
        self._cpu = cpu
    
    def __text_to_emb(self, text) -> np.array:
        if torch.cuda.is_available() and not self._cpu:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        tokenizer_from_file = self.bert_tokenizer
        tokenized = tokenizer_from_file(text,
                                        return_tensors="pt",
                                        max_length=512,
                                        add_special_tokens=True,
                                        truncation=True,
                                        )['input_ids'].numpy()
        tokenized = list(tokenized.reshape(-1))
        max_len = 200
        padded = np.array([tokenized + [0] * (max_len - len(tokenized))])
        attention_mask = np.where(padded != 0, 1, 0)
        input_ids = torch.tensor(padded).to(device)
        attention_mask = torch.tensor(attention_mask).to(device)
        m = self.bert_model.to(device)
        with torch.no_grad():
            last_hidden_states = m(input_ids, attention_mask=attention_mask)
        return last_hidden_states[0][:, 0, :].cpu().numpy()

    def predict(self, text_1: str, text_2: str) -> np.array:
        '''
        Method returns the cosine similarity between two texts
        
        '''
        return cosine_similarity(self.__text_to_emb(text_1), self.__text_to_emb(text_2))[0]