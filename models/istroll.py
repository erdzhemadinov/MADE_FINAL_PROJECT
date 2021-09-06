#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 19:42:55 2020

@author: Odegov Ilya, Zakladniy Anton
"""
import os
from copy import deepcopy
import pickle


from catboost import CatBoostError
from pandas import concat
from numpy import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


from models.lowlevelclassifiers import IsJoke


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


class FeatureExtractor:
    """
    Class extraxt features from question and answer.

    low-level-extractor:
        cos sim, toxic, joke, positive, threat, text enities, text stats
        and etc.

    @author: Odegov Ilya

    """

    def __init__(
            self, trainable, not_trainable, concat_trainable,
            low_level_extractors):
        """Init class object.

        Args:
            trainable (list): list of trainable feature extrctors.

            not_trainable (list): list of not trainable feature extrctors.

            concat_trainable (list): list of trainable feature extrctors,
            which get on input concat question and answer pair.

            low_level_extractors ([type]): list of feature extractor
            for low-level classifiers.

        """
        self._trainable = trainable
        self._not_trainable = not_trainable
        # feature extractors wich get concat question and answer as input data
        self._concat_trainable = concat_trainable
        self._low_level_extractors = low_level_extractors

    def save(self):
        """Save all low-level classifiers."""
        extractors = self._trainable + self._concat_trainable
        for extractor in extractors:
            extractor.save()

    def load(self):
        """Download from storage or train again low level model."""
        extractors = self._trainable + self._concat_trainable
        for extractor in extractors:
            print(f"Load {extractor.__class__}")
            extractor.load()

    def preprocess(self, datasets_features, labels):
        """Load from pre-trained or train low-level classifiers if file with save model doesn't exist."""
        extractors = self._trainable + self._concat_trainable
        extractors_and_data = zip(extractors , datasets_features, labels)
        for extractor, dataset_features, label in extractors_and_data:
            try:
                extractor.load()
                print(f"Succesful load {extractor.__class__}")
            except (FileNotFoundError, CatBoostError):
                print(f"Train {extractor.__class__}")
                extractor.fit(dataset_features, label)

    def train(self, datasets_features, labels):
        """Train all lower-level classifiers.

        Args:
            datasets_features (list): list of features for low-level classifiers.

            labels (list): list of target for low-level classifiers.

        """
        # train low-level classifier
        extractors = self._trainable + self._concat_trainable
        extractors_and_data = zip(extractors , datasets_features, labels)
        for extractor, dataset_features, label in extractors_and_data:
            print(f"Train {extractor.__class__}")
            extractor.fit(dataset_features, label)

    def _create_features(
            self, sequences, sequences_embedding):
        """Create features for questions or answers using text and their embedding.

        Args:
            sequences (str): question or answer in text format.

            sequences_embedding (ndarray): embeddings for question or answer.

        Returns:
            features (ndarrya): features for finall clasiifier from input text sequence.

        """
        features = []
        for low_level_classifier, extractor in zip(self._trainable, self._low_level_extractors):
            if extractor.feature_extractors:
                low_level_features = [sequences_embedding]
                temp = extractor.create_features(sequences)
                low_level_features.append(temp)
                low_level_features = hstack(low_level_features)
            else:
                low_level_features = sequences_embedding
            temp = low_level_classifier.predict_proba(low_level_features)
            features.append(temp)

        for extractor in self._not_trainable:
            features.append(extractor.create_features(sequences))
        return hstack(features)

    def create_features(
            self, questions, questions_embedding, answers=None, answers_embedding=None,
            questions_answers_embedding=None):
        """Create features for troll classifier.

        Use low-level classifiers for create features for final troll classifer.

        Args:
            questions (str, Series, list): questions.

            questions_embedding (ndarrya): embedding of questions.

            answers (str, Series, list): answers. Defaults to None.

            answers_embedding (ndarrya, optional): embedding of answers.
            Defaults to None.

            questions_answers_embedding (ndarray): embedding of concateated
            question-answer pairs. Defaults to None.

        Returns:
            ndarray: features for final troll classifier.

        """
        # get all features from questions
        questions_features = self._create_features(questions, questions_embedding)
        if all((answers is not None, answers_embedding is not None)):
            # get all features from answer
            answers_features = self._create_features(answers, answers_embedding)
            # compute similarity between question and answer pairs
            cosine = cosine_similarity(questions_embedding, answers_embedding)[0]
            # stack results
            result = hstack((
                questions_features, questions_embedding.mean(axis=1).reshape(-1, 1),
                answers_features, answers_embedding.mean(axis=1).reshape(-1, 1),
                cosine.reshape(-1, 1),
            ))
        else:
            result = hstack((
                questions_features, questions_embedding.mean(axis=1).reshape(-1, 1),
            ))
        if all((answers is not None, answers_embedding is not None, questions_answers_embedding is not None)):
            if self._concat_trainable:
                start = len(self._concat_trainable)
                classifiers_extractors = zip(
                    self._concat_trainable, self._low_level_extractors[-start:]
                )
                for low_level_classifier, extractor in classifiers_extractors:
                    if isinstance(low_level_classifier, IsJoke):
                        if isinstance(questions, str) and isinstance(answers, str):
                            text = "[SEP]".join((questions, answers))
                        else:
                            text = concat((questions, answers), axis=1).apply(
                                lambda row: "[SEP]".join((row[0], row[1]))
                            )
                        features = extractor.create_features(
                            text, questions_embedding, answers_embedding
                        )
                    else:
                        features = deepcopy(questions_answers_embedding)
                    low_level_features = low_level_classifier.predict_proba(features)
                    result = hstack((result, low_level_features))
        return result


class IsTroll:
    """
    Class implement high-level model for computing probability of trolling
    in pair question-answer.

    @author: Zakladniy Anton, Odegov Ilya

    """

    def __init__(
            self, penalty="l2", tol=1e-4, C=3., solver="sag", random_state=42,
            max_iter=1000):
        """Init class object

        Args:
            penalty (str, optional): penalty for troll classifier. Defaults to "l2".

            tol (float, optional): tol for troll classifier. Defaults to 1e-4.

            C (float, optional): C for troll classifier. Defaults to 3.

            solver (str, optional): solver for troll classifier. Defaults to "sag".

            random_state (int, optional): random state for troll classifier. Defaults to 42.

            max_iter (int, optional): max_iter for troll classifier. Defaults to 1000.

        """
        self._logreg = LogisticRegression(
            penalty=penalty, tol=tol, C=C, solver=solver,
            random_state=random_state, max_iter=max_iter, n_jobs=-1
        )
        self._scaler = StandardScaler()

    def save(self, save_path="save_models"):
        """Save model and scaler to pickle.

        Args:
            save_path (str, optional): path for save model and scaler.
            Defaults to "save_models".

        """
        save_path = os.path.join(CURRENT_DIR, save_path)
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "is_troll_model.pickle"), "wb") as fout:
            pickle.dump(self._logreg, fout)
        with open(os.path.join(save_path, "is_troll_scaler.pickle"), "wb") as fout:
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
            with open(os.path.join(load_path, "is_troll_model.pickle"), "rb") as fin:
                self._logreg = pickle.load(fin)
            with open(os.path.join(load_path, "is_troll_scaler.pickle"), "rb") as fin:
                self._scaler = pickle.load(fin)
        except FileNotFoundError as err:
            raise FileNotFoundError(err)

    def fit(self, x_data, y_data):
        """Fit model.

        Args:
            x_data (ndarray): input traind data.

            y_data (ndarray): input target data.

        """
        x_data = self._scaler.fit_transform(x_data)
        self._logreg.fit(X=x_data, y=y_data)

    def predict(self, x_data):
        """Predict class.

        Args:
            x_data (ndarray): test data.

        Returns:
            classes (float): resulted classes.

        """
        x_data = self._scaler.transform(x_data)
        classes = self._logreg.predict(X=x_data)
        return classes

    def predict_proba(self, x_data):
        """Predict probability of classes.

        Args:
            x_data (ndarray): test data.

        Returns:
            classes (float): resulted probbilities of troll.

        """
        x_data = self._scaler.transform(x_data)
        probs = self._logreg.predict_proba(X=x_data)
        return probs

    @property
    def logreg(self):
        """Getter classifier."""
        return self._logreg

    @property
    def scaler(self):
        """Getter data scaler."""
        return self._scaler
