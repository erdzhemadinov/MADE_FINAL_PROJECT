# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 13:20:40 2020

@author: User

"""

import pickle
import numpy as np
import torch
import pandas as pd
import os
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel, AutoConfig

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import model_selection
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import GridSearchCV

class BertToxicPredict():

    def __init__(self, bert_model, bert_tokenizer, classificator, cpu=False):
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer
        self.classificator = classificator
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

    def predict(self, text) -> np.array:
        '''
        Method returns the probability of belonging to a non-toxic class

        '''
        return np.array(self.classificator.predict_proba(self.__text_to_emb(text))[0][0])

    def get_predict_as_dataframe(self, text) -> pd.DataFrame:
        return pd.DataFrame({'prob_of_non_toxic': [self.predict(text)]})

    def fit(self,
            path_to_dataset="datasets/labeled.csv",
            save_model=True, path_save_model="logreg_new.pickle",
            save_emb_to_np=False, path_save_emb_to_np="emb_numpy/toxic.npy",
            load_emb=False):

        assert os.path.exists(path_to_dataset), 'dataset from '+path_to_dataset+' not found'
        data = pd.read_csv(path_to_dataset)

        if not load_emb:
            data_emb = []
            for i in tqdm(range(data.shape[0])):
                data_emb.append(self.__text_to_emb(data.iloc[i, 0]))

            data_emb_1 = np.stack(data_emb, axis=0)
            data_emb_1 = data_emb_1[:, -1, :]

        else:
            data_emb_1 = np.load(path_save_emb_to_np)

        if save_emb_to_np:
            np.save(path_save_emb_to_np, data_emb_1)

        X_train, X_test, y_train, y_test = model_selection.train_test_split(data_emb_1,
                                                                            data['toxic'].values,
                                                                            test_size=0.01,
                                                                            shuffle=True,
                                                                            random_state=42,
                                                                            )
        self.classificator = LogisticRegression(C=0.1)
        self.classificator.fit(X_train, y_train)

        y_train_pred = self.classificator.predict(X_train)
        y_test_pred = self.classificator.predict(X_test)

        print("Metrics toxic")
        print("ROC-AUC metric: train: {:.5f}, test: {:.5f}".format(roc_auc_score(y_train, y_train_pred), roc_auc_score(y_test, y_test_pred)))
        print("Accuracy metric: train: {:.5f}, test: {:.5f}".format(accuracy_score(y_train, y_train_pred), accuracy_score(y_test, y_test_pred)))
        print("F1-score metric: train: {:.5f}, test: {:.5f}".format(f1_score(y_train, y_train_pred), f1_score(y_test, y_test_pred)))

        if save_model:
            self.save_model(path_save_model)

    def save_model(self, model_filename = 'logreg_new.pickle'):
        pickle.dump(self.classificator, open(model_filename, 'wb'))
