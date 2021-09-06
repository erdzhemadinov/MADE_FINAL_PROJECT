#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 19:34:04 2020

@authors: Dzhemadinov Eskender, Ivanov Kirill, Kovtun Nikilay, Obidin Egor,
Odegov Ilya, Zakladniy Anton
"""
import os
import sys
from collections.abc import Iterable
import pickle
import re


from catboost import CatBoostClassifier
from numpy import asarray, hstack
from pytorch_lightning.metrics.functional.classification import auroc
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import cuda, nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset
from tqdm import tqdm


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(CURRENT_DIR)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


from utils import smart_batching


class IsToxic:
    """
    Class for define the toxic text.

    @author: Zakladniy Anton.

    """

    def __init__(self, penalty, tol, C, solver, random_state, max_iter):
        """Init class object."""
        self._scaler = StandardScaler()
        self._logreg = LogisticRegression(
            penalty=penalty, tol=tol, C=C, solver=solver,
            random_state=random_state, n_jobs=-1)

    def save(self, save_path="save_models"):
        """Save model and scaler to pickle.

        Args:
            save_path (str, optional): path for save model and scaler.
            Defaults to "save_models".

        """
        save_path = os.path.join(CURRENT_DIR, save_path)
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "is_toxic_model.pickle"), "wb") as fout:
            pickle.dump(self._logreg, fout)
        with open(os.path.join(save_path, "is_toxic_scaler.pickle"), "wb") as fout:
            pickle.dump(self._scaler, fout)

    def load(self, load_path="save_models"):
        """Load classifier model and scaler.

        Args:
            load_path (str, optional): path for load modelor scaler. Defaults to "save_models".

        Raises:
            FileNotFoundError: if file with save model does not exist.

        """
        load_path = os.path.join(CURRENT_DIR, load_path)
        try:
            with open(os.path.join(load_path, "is_toxic_model.pickle"), "rb") as fin:
                self._logreg = pickle.load(fin)
            with open(os.path.join(load_path, "is_toxic_scaler.pickle"), "rb") as fin:
                self._scaler = pickle.load(fin)
        except FileNotFoundError as err:
            raise FileNotFoundError(err)

    def fit(self, x_data, y_data):
        """Fit toxic classifier..

        Args:
            x_data (ndarray): input traind data.

            y_data (ndarray): input target data.

        """
        x_data = self._scaler.fit_transform(x_data)
        self._logreg.fit(X=x_data, y=y_data)

    def predict(self, x_data):
        """Predict toxic or not class.

        Args:
            x_data (ndarray): test data.

        Returns:
            classes (float): resulted classes.

        """
        x_data = self._scaler.transform(x_data)
        classes = self._logreg.predict(x_data)
        return classes

    def predict_proba(self, x_data):
        """Return probability what data is toxic."""
        x_data = self._scaler.transform(x_data)
        prob = self._logreg.predict_proba(x_data)[:, 1]
        return prob.reshape(-1, 1)


class ToxicFeatureExtractor:
    """
    Class for preprocess text and extract features for fit IsTroll classifier.

    @author: Odegov Ilya

    """

    def __init__(self, feature_extractors):
        """Init object class."""
        self._feature_extractors = feature_extractors

    def _preprocess_and_clean(self, text):
        """Clean and prepocess input text."""
        return text

    def create_features(self, text):
        """Compute text embeddings and other features."""
        features = []
        if self._feature_extractors:
            text = self._preprocess_and_clean(text)
            for extractor in self._feature_extractors:
                features.append(extractor.create_features(text))
            features = hstack(features)
        return features

    @property
    def feature_extractors(self):
        """Getter for extractor."""
        return self._feature_extractors


class IsPositive:
    """Classifier for sentiment.

    @author: Obidin Egor

    """

    def __init__(self, penalty, tol, C, solver, random_state, max_iter):
        """Init class object."""
        self._scaler = StandardScaler()
        self._logreg = LogisticRegression(
            penalty=penalty, tol=tol, C=C, solver=solver,
            random_state=random_state, max_iter=max_iter, n_jobs=-1)

    def save(self, save_path="save_models"):
        """Save model and scaler to pickle.

        Args:
            save_path (str, optional): path for save model and scaler.
            Defaults to "save_models".

        """
        save_path = os.path.join(CURRENT_DIR, save_path)
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "is_positive_model.pickle"), "wb") as fout:
            pickle.dump(self._logreg, fout)
        with open(os.path.join(save_path, "is_positive_scaler.pickle"), "wb") as fout:
            pickle.dump(self._scaler, fout)

    def load(self, load_path="save_models"):
        """Load classifier model and scaler.

        Args:
            load_path (str, optional): path for load modelor scaler. Defaults to "save_models".

        Raises:
            FileNotFoundError: if file with save model does not exist.

        """
        load_path = os.path.join(CURRENT_DIR, load_path)
        try:
            with open(os.path.join(load_path, "is_positive_model.pickle"), "rb") as fin:
                self._logreg = pickle.load(fin)
            with open(os.path.join(load_path, "is_positive_scaler.pickle"), "rb") as fin:
                self._scaler = pickle.load(fin)
        except FileNotFoundError as err:
            raise FileNotFoundError(err)

    def fit(self, x_data, y_data):
        """Fit positive classifier.

        Args:
            x_data (ndarray): input traind data.

            y_data (ndarray): input target data.

        """
        x_data = self._scaler.fit_transform(x_data)
        self._logreg.fit(X=x_data, y=y_data)

    def predict(self, x_data):
        """Predict positive or not class.

        Args:
            x_data (ndarray): test data.

        Returns:
            classes (float): resulted classes.

        """
        x_data = self._scaler.transform(x_data)
        classes = self._logreg.predict(x_data)
        return classes

    def predict_proba(self, x_data):
        """Get probability what x_data is positive."""
        x_data = self._scaler.transform(x_data)
        prob = self._logreg.predict_proba(x_data)
        return prob[:, 1].reshape(-1, 1) if prob.shape[1] == 2 else prob


class IsPositiveCatboost:
    """Classifier for sentiment.


    @author: Ivanov Kirill

    """
    def __init__(self, **kwargs):
        """Init class object."""
        self._clf = CatBoostClassifier(**kwargs)

    def save(self, save_path="save_models"):
        """Save model to pickle.

        Args:
            save_path (str, optional): path for save model and scaler.
            Defaults to "save_models".

        """
        save_path = os.path.join(CURRENT_DIR, save_path)
        os.makedirs(save_path, exist_ok=True)
        self._clf.save_model(
            os.path.join(save_path, "is_positive_model_catboost.cat")
        )

    def load(self, load_path="save_models"):
        """Load classifier model.

        Args:
            load_path (str, optional): path for load modelor scaler. Defaults to "save_models".

        Raises:
            FileNotFoundError: if file with save model does not exist.

        """
        load_path = os.path.join(CURRENT_DIR, load_path)
        self._clf.load_model(
            os.path.join(load_path, "is_positive_model_catboost.cat")
        )

    def fit(self, x_data, y_data, **kwargs):
        """Fit sentiment classifier..

        Args:
            x_data (ndarray): input traind data.

            y_data (ndarray): input target data.

        """
        self._clf.fit(X=x_data, y=y_data, **kwargs)

    def predict(self, x_data):
        """Predict positive or not class.

        Args:
            x_data (ndarray): test data.

        Returns:
            classes (float): resulted classes.

        """
        classes = self._clf.predict(x_data)
        return classes

    def predict_proba(self, x_data):
        """Get probability/probabilities what x_data is positive."""
        prob = self._clf.predict_proba(x_data)
        return prob[:, 1].reshape(-1, 1) if prob.shape[1] == 2 else prob


class PositiveFeatureExtractor:
    """
    Class for preprocess text and extract features for fit IsPositive classifier.

    @author: Obidin Egor

    """

    def __init__(self, feature_extractors=[]):
        """Init object class."""
        self._feature_extractors = feature_extractors

    def _preprocess_and_clean(self, text):
        """Clean and prepocess input text."""
        return text

    def create_features(self, text):
        """Compute text embeddings and other features."""
        features = []
        if self._feature_extractors:
            text = self._preprocess_and_clean(text)
            for extractor in self._feature_extractors:
                features.append(extractor.create_features(text))
            features = hstack(features)
        return features

    @property
    def feature_extractors(self):
        """Getter for extractor."""
        return self._feature_extractors


class IsThreat:
    """
    Class for predict is sequence a threat.

    @author: Kovtun Nikolay.

    """

    def __init__(self, iterations):
        """Class object init."""
        self._classifier = CatBoostClassifier(
            iterations=iterations, verbose=False
        )

    def save(self, save_path="save_models"):
        """Save model to pickle.

        Args:
            save_path (str, optional): path for save model and scaler.
            Defaults to "save_models".

        """
        save_path = os.path.join(CURRENT_DIR, save_path)
        self._classifier.save_model(
            os.path.join(save_path, "is_threat_model.cat")
        )

    def load(self, load_path="save_models"):
        """Load classifier model.

        Args:
            load_path (str, optional): path for load modelor scaler. Defaults to "save_models".

        Raises:
            FileNotFoundError: if file with save model does not exist.

        """
        load_path = os.path.join(CURRENT_DIR, load_path)
        self._classifier.load_model(
            os.path.join(load_path, "is_threat_model.cat")
        )

    def fit(self, x_data, y_data):
        """Fit threat classifier..

        Args:
            x_data (ndarray): input traind data.

            y_data (ndarray): input target data.

        """
        self._classifier.fit(x_data, y_data)

    def predict(self, x_data):
        """Predict threat or not class.

        Args:
            x_data (ndarray): test data.

        Returns:
            classes (float): resulted classes.

        """
        return self._classifier.predict(x_data)

    def predict_proba(self, x_data):
        """Predict probabilities of threat.

        Args:
            x_data (ndarray): test data.

        Returns:
            classes (float): resulted probabilities.

        """
        prob = self._classifier.predict_proba(x_data)[:, 1]
        return prob.reshape(-1, 1)


class ThreatFeatureExtractor:
    """
    Class for preprocess text and extract features for fit IsPositive classifier.

    @author: Odegov ILya

    """

    def __init__(self, feature_extractors=[]):
        """Init object class."""
        self._feature_extractors = feature_extractors
        self._tfidf = TfidfVectorizer(
            strip_accents='unicode', analyzer='char', ngram_range=(2, 3),
            max_features=2000
        )

    def _compute_tfidf(self, text):
        """Fit tf-idf and transform text to vetors."""
        return self._tfidf.fit_transform(text)

    def create_features(self, text):
        """Compute text embeddings and other features."""
        features = [self._compute_tfidf(text)]
        if self._feature_extractors:
            for extractor in self._feature_extractors:
                features.append(extractor.create_features(text))
            features = hstack(features)
        return features

    @property
    def feature_extractors(self):
        """Getter for extractor."""
        return self._feature_extractors


class IsJoke:
    """
    Class for define the joke.

    @author: Obidin Egor

    """

    def __init__(self, penalty, tol, C, solver, random_state, max_iter):
        """Init class object."""
        self._scaler = StandardScaler()
        self._logreg = LogisticRegression(
            penalty=penalty, tol=tol, C=C, solver=solver,
            random_state=random_state, max_iter=max_iter, n_jobs=-1)

    def save(self, save_path="save_models"):
        """Save model and scaler to pickle.

        Args:
            save_path (str, optional): path for save model and scaler.
            Defaults to "save_models".

        """
        save_path = os.path.join(CURRENT_DIR, save_path)
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "is_joke_model.pickle"), "wb") as fout:
            pickle.dump(self._logreg, fout)
        with open(os.path.join(save_path, "is_joke_scaler.pickle"), "wb") as fout:
            pickle.dump(self._scaler, fout)

    def load(self, load_path="save_models"):
        """Load classifier model and scaler.

        Args:
            load_path (str, optional): path for load modelor scaler. Defaults to "save_models".

        Raises:
            FileNotFoundError: if file with save model does not exist.

        """
        load_path = os.path.join(CURRENT_DIR, load_path)
        try:
            with open(os.path.join(load_path, "is_joke_model.pickle"), "rb") as fin:
                self._logreg = pickle.load(fin)
            with open(os.path.join(load_path, "is_joke_scaler.pickle"), "rb") as fin:
                self._scaler = pickle.load(fin)
        except FileNotFoundError as err:
            raise FileNotFoundError(err)

    def fit(self, x_data, y_data):
        """Fit joke classifier..

        Args:
            x_data (ndarray): input traind data.

            y_data (ndarray): input target data.

        """
        x_data = self._scaler.fit_transform(x_data)
        self._logreg.fit(X=x_data, y=y_data)

    def predict(self, x_data):
        """Predict joke or not class.

        Args:
            x_data (ndarray): test data.

        Returns:
            classes (float): resulted classes.

        """
        x_data = self._scaler.transform(x_data)
        prob = self._logreg.predict(x_data)
        return prob

    def predict_proba(self, x_data):
        """Predict probability what x_data is joke.

        Args:
            x_data (ndarray): test data.

        Returns:
            prob (float): resulted classes.

        """
        x_data = self._scaler.transform(x_data)
        prob = self._logreg.predict_proba(x_data)[:, 1]
        return prob.reshape(-1, 1)


class JokeFeatureExtractor:
    """
    Class for preprocess text and extract features for fit IsJoke classifier.

    @author: Obidin Egor

    """

    def __init__(self, feature_extractors=[]):
        """Init object class."""
        self._pca = PCA(n_components=5, random_state=42)
        self._feature_extractors = feature_extractors

    def split_text(self, text):
        """Split input data by SEP."""
        data = asarray(list(map(lambda x: x.split("[SEP]"), text)))
        return data[:, 0], data[:, 1]

    def _load_pca(self, load_path="save_models"):
        """Load PCA from pre-trained PCA."""
        load_path = os.path.join(CURRENT_DIR, load_path)
        with open(os.path.join(load_path, "is_joke_pca.pickle"), "rb") as fin:
            self._pca = pickle.load(fin)

    def _train_save_pca(self, features, save_path="save_models"):
        save_path = os.path.join(CURRENT_DIR, save_path)
        os.makedirs("save_models", exist_ok=True)
        self._pca.fit(features)
        with open(os.path.join(save_path, "is_joke_pca.pickle"), "wb") as fout:
            pickle.dump(self._pca, fout)

    def _compute_pca_transform(self, features):
        """Compute transform feature by load/train PCA model."""
        try:
            self._load_pca()
        except FileNotFoundError:
            self._train_save_pca(features)
        features = self._pca.transform(features)
        return features

    def create_features(self, text, questions_embeddings, answers_embeddings):
        """Compute text embeddings and other features."""
        features = hstack([questions_embeddings, answers_embeddings])
        # load/train pca
        features = [self._compute_pca_transform(features)]
        if self._feature_extractors:
            text = self._preprocess_and_clean(text)
            for extractor in self._feature_extractors:
                features.append(extractor.create_features(text))
        features = hstack(features)
        return features

    @property
    def feature_extractors(self):
        """Getter for extractor."""
        return self._feature_extractors


class IsUsefulOrSpam:
    """
    Classifier for useful or spam.

    @author: Obidin Egor

    """

    def __init__(self, penalty, tol, C, solver, random_state, max_iter):
        """Init class object."""
        self._scaler = StandardScaler()
        self._logreg = LogisticRegression(
            penalty=penalty, tol=tol, C=C, solver=solver,
            random_state=random_state, max_iter=max_iter, n_jobs=-1)

    def save(self, save_path="save_models"):
        """Save model and scaler to pickle."""
        save_path = os.path.join(CURRENT_DIR, save_path)
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "is_useful_from_spam_model.pickle"), "wb") as fout:
            pickle.dump(self._logreg, fout)
        with open(os.path.join(save_path, "is_useful_from_spam_scaler.pickle"), "wb") as fout:
            pickle.dump(self._scaler, fout)

    def load(self, load_path="save_models"):
        """Load classifier model and scaler.

        Args:
            load_path (str, optional): path for load modelor scaler. Defaults to "save_models".

        Raises:
            FileNotFoundError: if file with save model does not exist.

        """
        load_path = os.path.join(CURRENT_DIR, load_path)
        try:
            with open(os.path.join(load_path, "is_useful_from_spam_model.pickle"), "rb") as fin:
                self._logreg = pickle.load(fin)
            with open(os.path.join(load_path, "is_useful_from_spam_scaler.pickle"), "rb") as fin:
                self._scaler = pickle.load(fin)
        except FileNotFoundError as err:
            raise FileNotFoundError(err)

    def fit(self, x_data, y_data):
        """Fit usefulorspam classifier.

        Args:
            x_data (ndarray): input traind data.

            y_data (ndarray): input target data.

        """
        x_data = self._scaler.fit_transform(x_data)
        self._logreg.fit(X=x_data, y=y_data)

    def predict(self, x_data):
        """Predict classes useful or spam.

        Args:
            x_data (ndarray): test data.

        Returns:
            classes (float): resulted classes.

        """
        x_data = self._scaler.transform(x_data)
        prob = self._logreg.predict(x_data)
        return prob

    def predict_proba(self, x_data):
        """Predict probability what x_data is useful.

        Args:
            x_data (ndarray): test data.

        Returns:
            prob (float): resulted classes.

        """
        x_data = self._scaler.transform(x_data)
        prob = self._logreg.predict_proba(x_data)[:, 1]
        return prob.reshape(-1, 1)


class UsefulFeatureExtractor:
    """
    Class for preprocess text, extract features for IsUsefulOrSpam classifier.

    @author: Obidin Egor

    """

    def __init__(self, feature_extractors=[]):
        """Init object class."""
        self._feature_extractors = feature_extractors

    def split_text(self, text):
        """Split text by separator."""
        data = asarray(list(map(lambda x: x.split("[SEP]"), text)))
        return data[:, 0], data[:, 1]

    def create_features(self, text, questions_embeddings, answers_embeddings):
        """Compute text embeddings and other features."""
        features = [questions_embeddings, answers_embeddings]
        if self._feature_extractors:
            text = self._preprocess_and_clean(text)
            for extractor in self._feature_extractors:
                features.append(extractor.create_features(text))
        features = hstack(features)
        return features

    @property
    def feature_extractors(self):
        """Getter for extractor."""
        return self._feature_extractors


class IsBestClassifier(nn.Module):
    """
    Implement classifier for predict best answer. Pytorch version

    @author: Dzhemadinov Eskender

    """

    # define all the layers used in model
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        """Init class object."""
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.fc2 = nn.Linear(256, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.Sigmoid()

    def forward(self, inpt):
        """Forward step."""
        outputs = F.relu(self.fc1(inpt))
        outputs = F.relu(self.fc2(outputs))
        outputs = self.fc3(outputs)
        outputs = self.act(outputs)
        return outputs


class IsBest:
    """Class for train, predict, save and load IsBestClassifier.

    @author: Dzhemadinov Eskender

    """

    def __init__(
            self, batch_size_train, batch_size_val, epochs, learning_rate,
            device=None):
        """Init class object."""
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device
        # Set device
        self._get_dev()

    def _train(self):
        """Train model."""
        epoch_loss = 0
        epoch_acc = 0

        self.model.train()

        for batch in self.train_dataloader:
            self.optimizer.zero_grad()

            batch_ = batch[0].float().squeeze().unsqueeze(dim=0)
            # step of optimization
            predictions = self.model(batch_).squeeze()
            loss = self.criterion(predictions.float(), batch[1].float())
            acc = auroc(predictions.float(), batch[1].float())
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return (
            epoch_loss / len(self.train_dataloader),
            epoch_acc / len(self.train_dataloader)
        )

    def _evaluate(self):
        """Evaluate model."""
        epoch_loss = 0
        epoch_acc = 0

        self.model.eval()

        with torch.no_grad():
            for batch in self.test_dataloader:
                batch_ = batch[0].float().squeeze().unsqueeze(dim=0)
                predictions = self.model(batch_).squeeze()

                loss = self.criterion(predictions.float(), batch[1].float())
                acc = auroc(predictions.float(), batch[1].float())

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return (
            epoch_loss / len(self.test_dataloader),
            epoch_acc / len(self.test_dataloader)
        )

    def save(self, save_path="save_models"):
        """Save model after train.

        Args:
            save_path (str, optional): path for save model and scaler.
            Defaults to "save_models".

        """
        save_path = os.path.join(CURRENT_DIR, save_path)
        torch.save(
            self.model.state_dict(),
            os.path.join(save_path, "is_best.pt")
        )

    def load(self, load_path="save_models"):
        """Load classifier model and scaler.

        Args:
            load_path (str, optional): path for load modelor scaler. Defaults to "save_models".

        """
        load_path = os.path.join(CURRENT_DIR, load_path)
        embedding_dim = 768
        hidden_dim = 32
        output_dim = 1
        self.model = IsBestClassifier(embedding_dim, hidden_dim, output_dim)
        self.model.load_state_dict(
            torch.load(os.path.join(load_path, "is_best.pt"))
        )
        self.model.to(self.device)
        self.model.eval()

    def fit(self, x_data, y_data):
        """Fit isbest classifier.

        Fit calssifier and preprocess input data for satisfying format.

        Args:
            x_data (ndarray): input traind data.

            y_data (ndarray): input target data.

        """
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, train_size=0.85, random_state=42,
        )
        embedding_dim = x_train.shape[1]
        hidden_dim = 32
        output_dim = 1
        self.model = IsBestClassifier(embedding_dim, hidden_dim, output_dim)
        # preprocess test data and create test dataloader
        tensor_x = torch.as_tensor(x_train).to(self.device).float()
        tensor_y = torch.as_tensor(y_train.values).to(self.device).float()
        tensor_x.requires_grad = True

        train_dataset = TensorDataset(tensor_x, tensor_y)

        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size_train
        )
        # preprocess test data and create test dataloader
        tensor_x = torch.as_tensor(x_test).to(self.device).float()
        tensor_y = torch.as_tensor(y_test.values).to(self.device).float()

        train_dataset = TensorDataset(tensor_x, tensor_y)
        self.test_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size_val
        )

        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.BCELoss()

        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)

        best_valid_loss = float("inf")

        for epoch in range(self.epochs):

            train_loss, train_acc = self._train()

            valid_loss, valid_acc = self._evaluate()

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                self._best_model = self.model
            print(f"\tTrain Loss: {train_loss:.3f} | Train AUC: {train_acc * 100:.2f}")
            print(f"\t Val. Loss: {valid_loss:.3f} |  Val. AUC: {valid_acc * 100:.2f}")

    def _get_dev(self):
        """Get device information."""
        if self.device is not None:
            self.device = self.device
        else:
            self.device = 'cuda' if cuda.is_available() else 'cpu'

    def predict_proba(self, inpt):
        """Predict probability what x_data is best.

        Args:
            x_data (ndarray): test data.

        Returns:
            prob (float): resulted classes.

        """
        batch_size = inpt.shape
        prob = self.model(
            torch.as_tensor(inpt).to(self.device).view(batch_size[0], 1, -1)
        )
        return prob.cpu().detach().numpy().reshape(-1, 1)


class BestFeatureExtractor:
    """
    Class for preprocess text, extract features, create dataloader for IsBestClassifier.

    @author: Eskender Dzhemadinov

    """

    def __init__(self):
        """Init class object."""
        self._feature_extractors = None

    def create_features(self, text):
        """Nothing to do.

        Additional preprocess for input data is absent. All data preparation
        for train low-level calassifier locate inside it.

        """
        pass

    @property
    def feature_extractors(self):
        """Getter for extractor."""
        return self._feature_extractors


class TextStatistics():
    """Class for count text statistics as features for troll classificator.

    @author: Zkladniy Anton

    """

    def __init__(self):
        """Create class object."""
        self.stop_words = self._stop_words_loader()
        # pattern for url match
        self._regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\'.,<>?«»“”‘’]))"
        self._statistic_functions = [
            self._word_count, self._char_count, self._avg_word_len,
            self._stop_words_count, self._digit_count, self._title_word_count,
            self._url_count, self._emoticons_ratio_count
        ]

    def _stop_words_loader(self):
        """Load stop words."""
        with open(os.path.join(CURRENT_DIR, "stop_words.txt"), "r") as f:
            return f.read().splitlines()

    def _word_count(self, text) -> int:
        """Count quantity of words in the text."""
        return len(str(text).split(" "))

    def _char_count(self, text):
        """Count quantity of chars in the text."""
        return len(text)

    def _avg_word_len(self, text):
        """Count average word's length in the text."""
        words = text.split()
        return sum(len(word) for word in words) / len(words)

    def _stop_words_count(self, text):
        """Count quantity of stop words in the text."""
        return len([text for text in text.split() if text in self.stop_words])

    def _digit_count(self, text):
        """Count quantity of digit in the text."""
        return len([text for text in text.split() if text.isdigit()])

    def _title_word_count(self, text):
        """Count quantity of title words in the text."""
        return len([text for text in text.split() if text.istitle()])

    def _url_count(self, text):
        """Count quantity of url in the text."""
        url = re.findall(self._regex, text)
        return len([x[0] for x in url])

    def _emoticons_ratio_count(self, text):
        """Count good, bad emoticons."""
        pattern_good = r'\)'
        good = re.findall(pattern_good, text)

        pattern_bad = r'\('
        bad = re.findall(pattern_bad, text)

        if len(good) == len(bad):
            return 0
        if len(good) > len(bad):
            return 1
        if len(good) < len(bad):
            return -1

    def _create_features_for_single_sequence(self, sequence):
        """Create single sequence and compute all statistic."""
        stat_features = tuple([
            statistic_function(sequence)
            for statistic_function in self._statistic_functions
        ])
        return stat_features

    def _create_features_for_all_sequence(self, sequences):
        """Get all sequences and compute all statistic."""
        features = [
            self._create_features_for_single_sequence(sequence)
            for sequence in tqdm(sequences)
        ]
        return asarray(features)

    def create_features(self, text):
        """Create features for text or sequence of text."""
        if isinstance(text, str):
            return asarray(self._create_features_for_single_sequence(text)).reshape(1, -1)
        elif isinstance(text, Iterable):
            return self._create_features_for_all_sequence(text)
        else:
            raise TypeError("Input data has wrong type.")
