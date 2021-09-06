#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from pandas import read_csv
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier

from numpy import save, load
import torch
import torch.cuda
import pickle

from bertolet import Embedder, compute_metrics, print_metrics, binary_balance


class IsPositive:
    """Classifier for sentiment.

    """
    def __init__(self, penalty='l2', tol=0.0001, C=0.9,
                 solver='lbfgs', random_state=42, max_iter=100):
        """Init class object."""
        self._scaler = StandardScaler()
        self._logreg = LogisticRegression(
            penalty=penalty, tol=tol, C=C, solver=solver,
            random_state=random_state, max_iter=max_iter, n_jobs=-1)

    def save(self, save_path="save_models"):
        """Save model and scaler to pickle."""
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "is_positive_model.pickle"), "wb") as fout:
            pickle.dump(self._logreg, fout)
        with open(os.path.join(save_path, "is_positive_scaler.pickle"), "wb") as fout:
            pickle.dump(self._scaler, fout)

    def load(self, load_path="save_models"):
        """Load classifier model and scaler."""
        try:
            with open(os.path.join(load_path, "is_positive_model.pickle"), "rb") as fin:
                self._logreg = pickle.load(fin)
            with open(os.path.join(load_path, "is_positive_scaler.pickle"), "rb") as fin:
                self._scaler = pickle.load(fin)
        except FileNotFoundError as err:
            raise FileNotFoundError(err)

    def fit(self, x_data, y_data):
        """Train joke classifier."""
        x_data = self._scaler.fit_transform(x_data)
        self._logreg.fit(X=x_data, y=y_data)

    def predict(self, x_data):
        """Predict class."""
        x_data = self._scaler.transform(x_data)
        prob = self._logreg.predict(x_data)
        return prob

    def predict_proba(self, x_data):
        """Get probability what x_data is positive."""
        x_data = self._scaler.transform(x_data)
        prob = self._logreg.predict_proba(x_data)#[:, 1]
        return prob


class IsPositiveCatboost:
    """Classifier for sentiment.

    """
    def __init__(self, **kwargs):
        """Init class object."""
        self._clf = CatBoostClassifier(**kwargs)

    def save(self, save_path="save_models"):
        """Save model and scaler to pickle."""
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "is_positive_model_catboost.pickle"), "wb") as fout:
            pickle.dump(self._clf, fout)

    def load(self, load_path="save_models"):
        """Load classifier model and scaler."""
        try:
            with open(os.path.join(load_path, "is_positive_model_catboost.pickle"), "rb") as fin:
                self._clf = pickle.load(fin)
        except FileNotFoundError as err:
            raise FileNotFoundError(err)

    def fit(self, x_data, y_data, **kwargs):
        """Train joke classifier."""
        self._clf.fit(X=x_data, y=y_data, **kwargs)

    def predict(self, x_data):
        """Predict class."""
        prob = self._clf.predict(x_data)
        return prob

    def predict_proba(self, x_data):
        """Get probability what x_data is positive."""
        prob = self._clf.predict_proba(x_data)#[:, 1]
        return prob


if __name__ == "__main__":
    # dataset comments twitter (process_positive_negative.csv)
    # base model from BERT and logreg
    # output_size=50000; 
    # ____________________________________________
    # Embedder - BERT
    # Classifier - logreg
    # ROC-AUC ~ 0.75 (data cleared, no emoticons); 
    # ROC-AUC ~ 0.98 (data no cleared, there are emoticons)
    # ____________________________________________
    # Embedder - Dostoevsky
    # Classifier - logreg
    # ROC-AUC ~ 0.62 (data cleared, no emoticons); 
    # ROC-AUC ~ 0.85 (data no cleared, there are emoticons)
    # ____________________________________________
    # new dataset (process_comment_summary.csv)
    # output_size=None (~20000)
    # ____________________________________________
    # Embedder - BERT
    # Classifier - logreg
    # ROC-AUC metric:	train=0.80632, test=0.68828
    # Accuracy metric:	train=0.5727, test=0.51021
    # F1_score metric:	train=0.54899, test=0.48323
    # Precision metric:	train=0.56292, test=0.47299
    # Recall metric:	train=0.5727, test=0.51021
    # ____________________________________________
    # Embedder - BERT
    # Classifier - catboost
    # ROC-AUC: train=0.87262, test=0.80046 (data cleared, no emoticons); 
    
    # OUTPUT_SIZE = 50000
    TRAIN = False

    if TRAIN:
        test = True # True, если нужно собрать data embedding, False если он есть нужно загрузить его
        data = read_csv("datasets/process_comment_summary.csv")
        # data = binary_balance(
        #         data=read_csv("datasets/process_positive_negative.csv"),
        #         target="target",
        #         output_size=OUTPUT_SIZE,
        #         shuffle=True,
            # )
        target = data["target"]
        text = data["text"]
        
        if test:
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            embedder = Embedder(
                "DeepPavlov/rubert-base-cased-conversational",
                tokenizer="DeepPavlov/rubert-base-cased-conversational",
                device = device,
            )

            text_embeddings = embedder.encode(text)
            save('pos.npy', text_embeddings)
        else:
            text_embeddings = load('pos.npy')

        x_train, x_test, y_train, y_test = train_test_split(
            text_embeddings, target, train_size=0.85, random_state=42,
        )

        is_positive = IsPositiveCatboost()
        is_positive.fit(x_train, y_train)
        # metrics = compute_metrics(is_positive, x_train, y_train, x_test, y_test)
        metrics = compute_metrics(is_positive, x_train, y_train, x_test, y_test, bynary=False)
        print_metrics(metrics)

        y_predict = is_positive.predict(x_test)
        print(classification_report(y_test, y_predict))
        is_positive.save()

    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embedder = Embedder(
            "DeepPavlov/rubert-base-cased-conversational",
            tokenizer="DeepPavlov/rubert-base-cased-conversational",
            device = device,
        )

        is_positive = IsPositiveCatboost()
        is_positive.load()

        question = "Насколько этот вопрос позитивен?"
        answer = "Настолько насколько ты пойдешь на корм рыбам!"
        emb_q = embedder.encode([question])
        emb_a = embedder.encode([answer])
        print("\nquestion = {}\npredict = {}".format(question, is_positive.predict_proba(emb_q)[0]))
        print("\nanswer = {}\npredict = {}".format(answer, is_positive.predict_proba(emb_a)[0]))