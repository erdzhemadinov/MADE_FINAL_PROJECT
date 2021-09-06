#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 20:41:05 2020

@author: Odegov Ilya
"""
import os
import sys
from logging import getLogger, INFO, DEBUG
from time import sleep, time


from numpy import hstack, round_
from numpy import load as np_load
from numpy import save as np_save
from pandas import read_csv
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
from torch import cuda


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(CURRENT_DIR)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


from models.lowlevelclassifiers import (
    IsBest, IsJoke, IsPositiveCatboost, IsThreat, IsToxic
)
from models.lowlevelclassifiers import (
    BestFeatureExtractor, JokeFeatureExtractor, PositiveFeatureExtractor,
    ThreatFeatureExtractor, ToxicFeatureExtractor,
)
from models.lowlevelclassifiers import TextStatistics
from models.istroll import IsTroll, FeatureExtractor
from utils import Embedder, log, binary_balance


# features name wich extracted by low-level classifiers for summary dataset
TOKENIZER = "DeepPavlov/rubert-base-cased-conversational"
EMBEDDER = "DeepPavlov/rubert-base-cased-conversational"


def compute_metrics(model, x_train, y_train, x_test, y_test):
    """Get moodel and input data, compute metrics.

    Metrics: roc_auc

    Args:
        model (IsTroll): pretrained model
        x_train (ndarray): train data.

        y_train (ndarray): label for train data.

        x_test (ndarray): test input data.

        y_test (ndarray): label for test data.

    Returns:
        dict: dict of computed metrics

    """
    metrics = {}
    y_train_pred = model.predict_proba(x_train)
    y_test_pred = model.predict_proba(x_test)
    # dict of metric score functions
    metric_funcs = {"ROC-AUC": roc_auc_score}
    prec = 5
    for key, metric_func in metric_funcs.items():
        metrics[key] = {
            "train": round(metric_func(y_train, y_train_pred[:, 1]), prec),
            "test": round(metric_func(y_test, y_test_pred[:, 1]), prec)
        }
    return metrics


def print_metrics(metrics):
    """Print all metrics.

    Args:
        metrics (dict): dict of computed metrics.

    """
    for key, metric in metrics.items():
        print(f"{key} metric:\ttrain={metric['train']}, test={metric['test']}")


def _load_embedder(embedder_path, tokenizer):
    """Load bert and tokenizer, while create the object of Embedder class."""
    print("Create embedder. Load bert model and tokenizer.")
    embedder = os.path.join(CURRENT_DIR, embedder_path)
    device = torch.device("cuda:0" if cuda.is_available() else "cpu")
    embedder = Embedder(
        emberdder=embedder, tokenizer=tokenizer, device=device,
        tokenizer_max_length=512, clean=False
    )
    return embedder


def _read_csv(file_name, folder_path="datasets"):
    """Read csv from folder with hardcode params.

    Args:
        file_name (str): file name.

        folder_path (str, optional): folder for reading data. Defaults to "datasets".

    Returns:
        DataFrame: readed data.

    """
    folder_path = os.path.join(CURRENT_DIR, folder_path)
    data = read_csv(
        os.path.join(folder_path, file_name), names=["text", "target"],
        header=0, dtype={"text": "str", "target": "int"}
    )
    return data


def _create_train_data(datasets_name):
    """Read, preprocess datasets for low-level.

    Args:
        dataset_name ([type]): list of name of dataset for
        train low-level classifier.

    Returns:
        (DataFrame, Series): input data for classifier (text and target).

    """
    print("Read support datasets.")
    datasets = [_read_csv(name).iloc[:] for name in tqdm(datasets_name)]
    # create texts and targets dataset for support datasets.
    texts = [dataset["text"] for dataset in datasets]
    targets = [dataset["target"] for dataset in datasets]
    return texts, targets


def _load_or_create_features_for_low_level_classifier(
        embedder, low_level_extractors, texts, dataset_features_names,
        is_load=True, is_save=True):
    """Load or create new train data for low-level classifiers.

    Args:
        embedder (Embedder): model for creating text embeddings.

        low_level_extractors (list): list of feature extractors
        for low-level classsifier.

        texts (list): list with data for low-level classifier.

        dataset_features_names (list): list of name for trace output.

        is_load (bool, optional): flag responsible for load or train.
            Defaults to True.

        is_save (bool, optional): flag responsible for save getting
        results. Defaults to True.

    Returns:
        list(ndarray): list of features for low-level calssifier.

    """
    if is_load:
        print("Load/create features for support datasets.")
    else:
        print("Create features for support datasets.")
    extractors_texts_and_names = zip(
        low_level_extractors, texts, dataset_features_names
    )
    datasets_features = []
    path = os.path.join(CURRENT_DIR, "features")
    for extractor, text, name in extractors_texts_and_names:
        features_path = os.path.join(path, name)
        try:
            # load features matrix
            if is_load:
                dataset_features = np_load(features_path)
                print(f"Load {name} features.")
            else:
                raise FileNotFoundError()
        except FileNotFoundError:
            # if file does not eist create embeddings
            print(f"Create {name} features.")
            if isinstance(extractor, JokeFeatureExtractor):
                questions, answers = extractor.split_text(text)
                questions_embeddings = embedder.encode(questions)
                answers_embeddings = embedder.encode(answers)
                dataset_features = extractor.create_features(
                    text, questions_embeddings, answers_embeddings
                )
            else:
                embeddings = embedder.encode(text)
                if extractor.feature_extractors:
                    temp = extractor.create_features(text)
                    dataset_features = hstack([embeddings, temp])
                else:
                    dataset_features = embeddings
            # save create features matrixes
            if is_save:
                np_save(features_path, dataset_features)
        datasets_features.append(dataset_features)

    return datasets_features


def _create_troll_features_extractor(low_level_extractors):
    """Create the object of FeatureExtractor, load all pretrained low-level classifiers.

    Args:
        low_level_extractors (list): list of feature extractors for low-level
        classifiers.

    Returns:
        FeatureExtractor: feature extractor for final troll classifier.

    """
    trainable = [
        IsToxic(
            penalty="l2", tol=1e-4, C=0.5, solver="saga", random_state=42,
            max_iter=1000
        ),
        IsPositiveCatboost(verbose=False),
        IsThreat(iterations=1000),
    ]
    not_trainable = [TextStatistics()]
    concat_trainable = [
        IsJoke(
            penalty='l2', tol=1e-4, C=0.5, solver='saga', random_state=42,
            max_iter=1000
        ),
        IsBest(
            batch_size_train=1024, batch_size_val=1024, epochs=20,
            learning_rate=3e-4
        )
    ]
    extractor = FeatureExtractor(
        trainable=trainable, not_trainable=not_trainable,
        concat_trainable=concat_trainable,
        low_level_extractors=low_level_extractors,
    )
    return extractor


def _load_or_fit_troll_features_extractor(
        extractor, datasets_features, targets, is_load=True):
    """Load/train troll feature extractor.

    Args:
        extractor (FeatureExtractor): does not fit feature extractor
        for troll classifier.

        datasets_features (list): list  of features for low-level classifiers.

        targets ([type]): list of target for low-level classifiers.

        is_load (bool, optional): [description]. Defaults to True.

    Returns:
        FeatureExtractor: fited feature extractor for troll classifier.

    """
    # Train model
    print("Fit/Load low-level classifiers.")
    start = time()
    if is_load:
        extractor.preprocess(datasets_features, targets)
    else:
        extractor.train(datasets_features, targets)
    stop = time()
    mins, secs = (stop - start) // 60, int((stop - start) % 60)
    print(f"Train time: {mins} min {secs} sec.")
    return extractor


class IsTrollClassifierModel:
    """Class for process data, train or load and predict."""

    def __init__(self, embedder="bert", train_dataset="troll_data_big.csv"): # troll_data_big.csv troll_data_all.csv
        """Init class object.

        Args:
            embedder (str, BertModel): path to save model, model name
            on huggingface.co or preloaded Bert model. Defaults to "bert".

            train_dataset (str, optional): dataset for train finall classifier.
            Defaults to "troll_data_big.csv".

        """
        self._loger = getLogger("logger.IsTrollClassifierModel")
        self._train_dataset = train_dataset
        self._embedder = self._load_embedder(embedder)
        self._datasets_name = [
            "process_big_toxic_balance.csv", "process_comment_summary.csv",
            "process_threat.csv", "process_joke.csv", "process_best.csv"
        ]
        self._dataset_features_names = [
            "toxic_features.npy", "positive_features.npy",
            "threat_features.npy", "joke_features.npy",
            "best_answer_features.npy"
        ]
        self._low_level_extractors = [
            ToxicFeatureExtractor([TextStatistics()]),
            PositiveFeatureExtractor([TextStatistics()]),
            ThreatFeatureExtractor(), JokeFeatureExtractor(),
            BestFeatureExtractor()
        ]
        self._extractor = _create_troll_features_extractor(
            self._low_level_extractors
        )
        self._is_troll = IsTroll(
            penalty="l2", tol=1e-4, C=3, solver="sag", random_state=42,
            max_iter=1000
        )

    def _load_embedder(self, embedder, tokenizer=TOKENIZER):
        """Loader for embedder.

        Args:
            embedder (str, BertModel): path to save model,
            model name on huggingface.co or preloaded Bert model.

            tokenizer (str): tokenizer name on huggingface.co. Defaults to TOKENIZER.

        Returns:
            embedder (Embedder): model for embedding text.

        """
        if embedder != EMBEDDER and isinstance(embedder, str):
            embedder = _load_embedder(embedder, tokenizer)
        else:
            device = torch.device("cuda:0" if cuda.is_available() else "cpu")
            # device = torch.device("cpu")
            embedder = Embedder(
                emberdder=embedder, tokenizer=tokenizer, device=device,
                tokenizer_max_length=512, clean=True
            )
        return embedder

    def _load_or_create_embeddings(self, text, path, name, is_load, save):
        """Load or create new embeddings of input text.

        Args:
            text (Dataframe): Dataframe with text.

            path (Series): path flor load or save result.

            name (str): file name for load or save result.

            is_load (bool, optional): flag responsible for load or train.
            Defaults to True.

            is_save (bool, optional): flag responsible for save getting
            results. Defaults to True.

        Returns:
            ndarray: embeddings of text.

        """
        path = os.path.join(path, name)
        try:
            if is_load:
                embeddings = np_load(path)
            else:
                raise FileNotFoundError()
        except FileNotFoundError:
            embeddings = self._embedder.encode(text)
        if save:
            np_save(path, embeddings)
        return embeddings

    def _load_of_fit_features_extractor(
            self, texts, targets, is_load=True, is_save=True):
        """Prepare all support data, create classifier, fit/load.

        Args:
            texts (list(Dataframe)): list of support Dataframes for low-level calassifier.

            targets (list(Series)): list of target Series for low-level calassifier.

            is_load (bool, optional): flag responsible for load or train.
            Defaults to True.

            is_save (bool, optional): flag responsible for save getting
            results. Defaults to True.

        """
        datasets_features = _load_or_create_features_for_low_level_classifier(
            self._embedder, self._low_level_extractors, texts,
            self._dataset_features_names, is_load, is_save
        )
        #########################################################
        print("Create FeatureExtractor object.")
        self._extractor = _load_or_fit_troll_features_extractor(
            self._extractor, datasets_features, targets, is_load
        )

    def _load_or_create_embeddings_troll_dataset(
            self, troll, is_load=True, is_save=True):
        """Preprocess data, load/create embeddings.

        Args:
            troll (Dataframe): Dataframe with text pair question-answer
            and classification target.

            is_load (bool, optional): flag responsible for load or train.
            Defaults to True.

            is_save (bool, optional): flag responsible for save getting
            results. Defaults to True.

        Returns:
            [ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ]: result of process troll dataset

        """

        troll = binary_balance(troll, "trollolo")
        print("Create/load embeddings for main troll datasets.")
        sleep(0.2)
        questions = troll["question"]
        answers = troll["answer"]
        questions_answers = troll[["question", "answer"]].apply(
            lambda row: " ".join(row), axis=1
        )
        targets = troll["trollolo"].astype(int)
        if is_save:
            targets.to_csv(
                os.path.join(CURRENT_DIR, "datasets", "balance_troll.csv"),
                index=None
            )
        # save embeddings
        embedding_save_directory = os.path.join(CURRENT_DIR, "embeddings")
        q_embeddings = self._load_or_create_embeddings(
            questions, embedding_save_directory, "q_embeddings.npy",
            is_load, is_save
        )
        a_embeddings = self._load_or_create_embeddings(
            questions, embedding_save_directory, "a_embeddings.npy",
            is_load, is_save
        )
        qa_embeddings = self._load_or_create_embeddings(
            questions_answers, embedding_save_directory, "qa_embeddings.npy",
            is_load, is_save
        )
        return targets, questions, q_embeddings, answers, a_embeddings, questions_answers, qa_embeddings

    def _fit_troll_classiier(self, features, targets):
        """Train main troll classifier.

        Split data on test/train. Train classifier. Compute metric.
        Re-train classifier on full input data.

        Args:
            features ndarray: array of input data.

            targets ndarray: array of label.

        Returns:
            float: computed metric.
        """
        # start train troll classifier.
        x_train, x_test, y_train, y_test = train_test_split(
            features, targets, train_size=0.85, random_state=42,
        )
        print("Train troll classifier on split train data.")
        self._is_troll.fit(x_train, y_train)
        y_predict = self._is_troll.predict(x_test)
        print(classification_report(y_test, y_predict))
        metrics = compute_metrics(
            self._is_troll, x_train, y_train, x_test, y_test
        )
        print_metrics(metrics)
        print("Train troll classifier on all data.")
        self._is_troll.fit(features, targets)
        return metrics

    def save(self):
        """Save main is_troll model, and low-level calassifiers."""
        self._extractor.save()
        self._is_troll.save()

    def load(self):
        """load main is_troll model, and low-level calassifiers."""
        self._extractor.load()
        self._is_troll.load()

    def load_or_fit(self):
        """
        Train/load all classifier.

        Create/load embedder, low-level featurters extractors and classifiers,
        main model. Train/load all classifiers and validate troll classifier.

        Returns:
            float: computed metric.

        """
        texts, targets = _create_train_data(self._datasets_name)
        features_directory = os.path.join(CURRENT_DIR, "features")
        os.makedirs(features_directory, exist_ok=True)
        self._load_of_fit_features_extractor(texts, targets)
        #########################################################
        print("Read main troll dataset.")
        troll = read_csv(
            os.path.join(CURRENT_DIR, "datasets", self._train_dataset)
        )
        embedding_save_directory = os.path.join(CURRENT_DIR, "embeddings")
        os.makedirs(embedding_save_directory, exist_ok=True)
        result = self._load_or_create_embeddings_troll_dataset(troll)
        targets, questions, q_embeddings, answers, a_embeddings, questions_answers, qa_embeddings = result
        print("Extract features from questions and answers.")
        sleep(0.2)
        features = self._extractor.create_features(
            questions, q_embeddings, answers, a_embeddings, qa_embeddings
        )
        np_save(
            os.path.join(features_directory, "troll_features.npy"), features
        )
        metrics = self._fit_troll_classiier(features, targets)
        return metrics

    def fit(self):
        """
        Train model.

        Create embedder, low-level featurters extractors and classifiers,
        main model. Train all classifiers and validate troll classifier.

        Returns:
            float: computed metric.

        """
        texts, targets = _create_train_data(self._datasets_name)
        features_directory = os.path.join(CURRENT_DIR, "features")
        os.makedirs(features_directory, exist_ok=True)
        self._load_of_fit_features_extractor(
            texts, targets, is_load=False, is_save=True
        )
        #########################################################
        print("Read main troll dataset.")
        troll = read_csv(
            os.path.join(CURRENT_DIR, "datasets", self._train_dataset)
        )
        targets = troll["trollolo"].astype(int)
        result = self._load_or_create_embeddings_troll_dataset(troll, is_load=False, is_save=False)
        targets, questions, q_embeddings, answers, a_embeddings, questions_answers, qa_embeddings = result
        print("Extract features from questions and answers.")
        sleep(0.2)
        features = self._extractor.create_features(
            questions, q_embeddings, answers, a_embeddings, qa_embeddings
        )
        np_save("./features/troll_features.npy", features)
        metrics = self._fit_troll_classiier(features, targets)
        return metrics["ROC-AUC"]["test"]


    def predict(self, question, answer):
        """
        Compute probability of troll for input pair question-answer.

        Args:
            question (str): question part of pair.

            answer (str): answer part of pair.

        Returns:
            float: probability of troll.

        """
        q_embedding = self._embedder.encode(question)
        a_embedding = self._embedder.encode(answer)
        question_answer = " ".join((question, answer))
        qa_embedding = self._embedder.encode(question_answer)
        self._extractor.load()
        self._is_troll.load()
        features = self._extractor.create_features(
            question, q_embedding, answer, a_embedding, qa_embedding
        )
        return round_(self._is_troll.predict_proba(features)[0][1], 3)

    def create_features_for_single_text_sequence(self, sequence):
        """
        Create features for input text by use low-level extractor.

        Args:
            sequence (str): text sequence for preprocess

        Returns:
            features ndarray: features from low-level classifier.

        """
        s_embedding = self._embedder.encode(sequence)
        self._extractor.load()
        self._is_troll.load()
        features = self._extractor.create_features(sequence, s_embedding)
        return features

    @property
    def embedder(self):
        """Getter for embedder."""
        return self._embedder

    @property
    def extractor(self):
        """Getter for extractor."""
        return self._extractor

    @property
    def is_troll(self):
        """Getter for main troll model."""
        return self._is_troll


if __name__ == "__main__":
    obj = IsTrollClassifierModel()
    # obj.load_or_fit()
    # obj.save()
    print(obj.fit())
    obj.save()
    # print(obj.predict("Сколько весит Земля?", "Вес Земли примерно 1000000 тонн"))
