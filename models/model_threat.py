import os

from pandas import read_csv
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import torch

from bertolet import Embedder, compute_metrics, print_metrics
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
import numpy as np

class TfIdf():
    def __init__(self):
        self._char_vectorizer = TfidfVectorizer(strip_accents='unicode',
                                                analyzer='char',
                                                ngram_range=(2, 3),
                                                max_features=2000)
    def fit_transform(self, sentences):
        return self._char_vectorizer.fit(sentences).transform(sentences).toarray()

class IsClassifier:
    def __init__(self, *args, **kwargs):
        self._classifier = CatBoostClassifier(iterations = 1000)


    def save(self, save_path="save_models"):
        os.makedirs(save_path + '/threat', exist_ok=True)
        self._classifier.save_model(save_path + '/threat/threat')

    def load(self, load_path="save_models"):
        self._classifier.load_model(load_path + '/threat/threat')

    def fit(self, x_data, y_data):
        self._classifier.fit(x_data, y_data)

    def predict(self, x_data):
        return self._classifier.predict(x_data)

    def predict_proba(self, x_data):
        return self._classifier.predict_proba(x_data)[:, 1]

if __name__ == "__main__":
    data = read_csv("process_threat.csv")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    embedder = Embedder(
        emberdder="DeepPavlov/rubert-base-cased-conversational",
        tokenizer="DeepPavlov/rubert-base-cased-conversational",
        device = device, tokenizer_max_length=512, clean=False
    )

    target = data["target"]
    text = data["text"]
    text_embeddings = embedder.encode(text)
    text_embeddings = np.append(text_embeddings, TfIdf().fit_transform(data["text"].values), axis = 1)

    x_train, x_test, y_train, y_test = train_test_split(
        text_embeddings, target, train_size=0.85, random_state=42,
    )
    classifier = IsClassifier()
    classifier.fit(x_train, y_train)
    y_predict = classifier.predict(x_test)
    metrics = compute_metrics(classifier, x_train, y_train, x_test, y_test)
    print_metrics(metrics)
    print(classification_report(y_test, y_predict))
    print("ROC-AUC metric on predict_proba: " + str(roc_auc_score(y_test, classifier.predict_proba(x_test))))
