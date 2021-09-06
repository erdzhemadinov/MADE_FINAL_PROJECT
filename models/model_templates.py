#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 20:48:44 2020

@author: Odegov Ilya
"""
import os

from pandas import read_csv
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import torch
import torch.cuda

from bertolet import Embedder, compute_metrics, print_metrics

"""
Шаблонныый скрипт, в котором описывается основной функционал нижнеуовневого
классификатора, будь то токсичность, угрозы, луччший ответ, тональность
и так далее. В нем будет представлена примерная схема получения фичи для
верхнеуровнего троллингового классификатора. Этот шаблон можно брать за основу.
И менять под свои модели. Основные методы которые должны быть будут помечены
при прмощи raise NotImplementedError.

"""


"""
Модуль bertolet расположен в папке models. Которая в свою очередь расположена
в корневой директории проекта.

Класс Embredder принимает на вход путь до модели берта и до токенизатора
(путь до модели и токенизатора может быть локальной дирректорией или путь
с huggingface.co).

embedder = Embedder(
    embedder="DeepPavlov/rubert-base-cased-conversational",
    tokenizer="DeepPavlov/rubert-base-cased-conversational",
    device=зависит от того есть у вас GPU или нет,
    tokenizer_max_length=максимальное количество токенов,
        которые беруться из последовательности,
    clean=флаг для того, нужно проводить очистку текста или нет.
    )

"""

"""
Ваш класс, отвечающий за классификацию.

Обязательные к реализации методы:
    -fit;
    -prefict;
    -prefict_proba;
    -save;
    -load;

"""
class IsClassifier:
    def __init__(self, *args, **kwargs):
        """Принимаат на вход нужно количество параметров."""
        self._scaler = "ваш нормализатор данных."
        self._classifier = "ваш кллассификатор."


    def save(self, save_path="save_models"):
        """
        Сохраняет параметры модели и нормализатора.

        Папка по умолчанию не изменяется.
        """
        os.makedirs("save_path", exist_ok=True)
        raise NotImplementedError

    def load(self, load_path="save_models"):
        """Загружает сохранненные параметры модели и нормализатора."""
        raise NotImplementedError

    def fit(self, x_data, y_data):
        """Получает на вход вектор признаков номализуе их и обучает модель."""
        raise NotImplementedError

    def predict(self, x_data):
        """Получает на вход вектор признаков, нормализует, предсказывает класс."""
        raise NotImplementedError

     def predict_proba(self, x_data):
        """Получает на вход вектор признаков, нормализует, предсказывает принадлежность классу."""
        raise NotImplementedError



if __name__ == "__main__":
    """
    Стандартный pipeline обучения и предсказания.
    Чтение данных
    """
    data = read_csv("path_to_dataset.csv")
    device = torch.device("cuda:0" if cuda.is_available() else "cpu")
    embedder = Embedder(
        embedder=="путь до папки с моделью или DeepPavlov/rubert-base-cased-conversational",
        tokenizer="путь до папки с токенайзером или DeepPavlov/rubert-base-cased-conversational",
        device = device, tokenizer_max_length="число от 1 до 512"
    )
    """
    Прошу от вас ЕДИНООБРАЗИЯ тренировочных датасетов.
    Весь текст в колонке text, все метки в колонке target.
    """
    target = data["target"]
    text = data["texr"]
    text_embeddings = embedder.encode(text)
    """
    Возможно, помимо построения эбеддингов у вас будет дополнительная
    обратботка текска. Просьба выносить ее в отдельных класс.
    Дальше уже будем думать, как ее внедрять в финальную модель и оцениавать
    ее эффективность на всей модели в общем и целом.

    """
    x_train, x_test, y_train, y_test = train_test_split(
        text_embeddings, target, train_size=0.85, random_state=42,
    )
    classifier = IsClassifier()
    classifier.fit(x_train, y_train)
    y_predict = classifier.predict(x_test)
    metrics = compute_metrics(classifier, x_train, y_train, x_test, y_test)
    print_metrics(metrics)
    print(classification_report(y_test, y_predict))



"""
ОБЯЗАТЕЛЬНЫМ УСЛОВИЕМ является документирование кода, в котором реализованы
не тривиальные вещи. Так же необходимо указать НАБОР БИБЛИОТЕК, которые
использовались и которые НЕ УКАЗАНЫ В import шаблона.

"""




'''
Пример реализации класса классификации токсичности.

class IsToxic:
    """Class define toxic text.

    author Zakladniy Anton.

    """

    def __init__(self, penalty, tol, C, solver, random_state, max_iter):
        """Init class object."""
        self._scaler = StandardScaler()
        self._logreg = LogisticRegression(
            penalty=penalty, tol=tol, C=C, solver=solver,
            random_state=random_state, n_jobs=-1)

    def save(self, save_path="save_models"):
        """Save model and scaler to pickle."""
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "is_toxic_model.pickle"), "wb") as fout:
            pickle.dump(self._logreg, fout)
        with open(os.path.join(save_path, "is_toxic_scaler.pickle"), "wb") as fout:
            pickle.dump(self._scaler, fout)

    def load(self, load_path="save_models"):
        """Load classifier moel and scaler."""
        try:
            with open(os.path.join(load_path, "is_toxic_model.pickle"), "rb") as fin:
                self._logreg = pickle.load(fin)
            with open(os.path.join(load_path, "is_toxic_scaler.pickle"), "rb") as fin:
                self._scaler = pickle.load(fin)
        except FileNotFoundError as err:
            raise FileNotFoundError(err)

    def fit(self, x_data, y_data):
        """Train toxic classifier."""
        x_data = self._scaler.fit_transform(x_data)
        self._logreg.fit(X=x_data, y=y_data)

    def predict(self, x_data):
        """Get probability what x_data is toxic."""
        x_data = self._scaler.transform(x_data)
        prob = self._logreg.predict(x_data)
        return prob

    def predict(self, x_data):
        """Return class."""
        x_data = self._scaler.transform(x_data)
        prob = self._logreg.predict(x_data)
        return prob

    def predict_proba(self, x_data):
        """Return probability what data is toxic."""
        x_data = self._scaler.transform(x_data)
        prob = self._logreg.predict_proba(x_data)[:, 1]
        return prob
'''
