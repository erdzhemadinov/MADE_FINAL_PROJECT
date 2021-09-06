#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 22:28:32 2020

@author: Obidin Egor
"""
import os

from pandas import read_csv
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from numpy import array, hstack, save, load
import torch
import torch.cuda
import pickle

from bertolet import Embedder, compute_metrics, print_metrics

class IsJoke:
    """Classifier for joke.

    @author: Obidin Egor
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
        with open(os.path.join(save_path, "is_joke_model.pickle"), "wb") as fout:
            pickle.dump(self._logreg, fout)
        with open(os.path.join(save_path, "is_joke_scaler.pickle"), "wb") as fout:
            pickle.dump(self._scaler, fout)

    def load(self, load_path="save_models"):
        """Load classifier model and scaler."""
        try:
            with open(os.path.join(load_path, "is_joke_model.pickle"), "rb") as fin:
                self._logreg = pickle.load(fin)
            with open(os.path.join(load_path, "is_joke_scaler.pickle"), "rb") as fin:
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
        """Get probability what x_data is joke."""
        x_data = self._scaler.transform(x_data)
        prob = self._logreg.predict_proba(x_data)[:, 1]
        return prob

class JokeFeatureExtractor:
    """Class for preprocess text and extract features for fit IsJoke classifier."""

    def __init__(self):
        """Init object class"""
        self._other_feature_extractors = []

    def split_text(self, text):
        data = array(list(map(lambda x: x.split("[SEP]"), text)))
        return data[:,0], data[:,1]

    def create_features(self, text, emb_question, emb_answer):
        """Compute text embeddings and other features."""
        features = hstack((emb_question, emb_answer))
        return features


if __name__ == "__main__":

    TRAIN = True

    if TRAIN:
        test = True # True, если нужно собрать data embedding, False если он есть нужно загрузить его
        if test:
            data = read_csv("datasets/process_joke.csv")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            embedder = Embedder(
                "DeepPavlov/rubert-base-cased-conversational",
                tokenizer="DeepPavlov/rubert-base-cased-conversational",
                device = device, clean=False,
            )

            feature_extractor = JokeFeatureExtractor(embedder)

            target = data["target"]
            text = data["text"]

            question, answer = feature_extractor.split_text(text)
            text_embeddings = feature_extractor.create_features(text,
                                                                embedder.encode(question),
                                                                embedder.encode(answer))

            save('ll.npy', text_embeddings)
        else:
            text_embeddings = load('ll.npy')
            data = read_csv("datasets/process_joke.csv")
            target = data["target"]

        x_train, x_test, y_train, y_test = train_test_split(
            text_embeddings, target, train_size=0.85, random_state=42,
        )

        is_joke = IsJoke()
        is_joke.fit(x_train, y_train)
        metrics = compute_metrics(is_joke, x_train, y_train, x_test, y_test)
        print_metrics(metrics)

        y_predict = is_joke.predict(x_test)
        print(classification_report(y_test, y_predict))
        is_joke.save()

    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embedder = Embedder(
            "DeepPavlov/rubert-base-cased-conversational",
            tokenizer="DeepPavlov/rubert-base-cased-conversational",
            device = device,
        )

        feature_extractor = JokeFeatureExtractor(embedder)

        is_joke = IsJoke()
        is_joke.load()

        question = "Почему приняты за единицу расстояния между телами в Солнечной системе 150 млн. километров?"
        answer = "Потому что это расстояние от Земли до Солнца"
        emb_qa = feature_extractor.create_features('',
                                                   embedder.encode([question]),
                                                   embedder.encode([answer]))
        print("\nquestion = {}\nanswer = {}\npredict = {}".format(question, answer, is_joke.predict_proba(emb_qa)[0]))

        question = "Какая разница между падением в 10-метровую яму и 100-метровую пропасть?"
        answer = "Крышка гроба будет открытой или закрытой."
        emb_joke = feature_extractor.create_features('',
                                                     embedder.encode([question]),
                                                     embedder.encode([answer]))
        print("\nquestion = {}\nanswer = {}\npredict = {}".format(question, answer, is_joke.predict_proba(emb_joke)[0]))


