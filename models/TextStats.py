# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import re


class TextStatistics():
    '''
    Class for count text statistics as features for classificator

    '''

    def __init__(self, stop_words):
        self.stop_words = stop_words

    # def stop_words_loader(self):
    #     with open('stop_words.txt', 'r') as f:
    #         return f.read().splitlines()

    def __word_count(self, text) -> int:
        return len(str(text).split(' '))

    def __char_count(self, text) -> int:
        return len(text)

    def __avg_word_count(self, text) -> int:
        words = text.split()
        return (sum(len(word) for word in words) / len(words))

    def __stop_words_coun(self, text) -> int:
        return len([text for text in text.split() if \
                    text in self.stop_words])

    def __numbers_count(self, text) -> int:
        return len([text for text in text.split() if \
                    text.isdigit()])

    def __title_word_count(self, text) -> int:
        return len([text for text in text.split() if \
                    text.istitle()])

    def __url_count(self, text) -> int:
        regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
        url = re.findall(regex, text)
        return len([x[0] for x in url])

    def __bracket_mood_check(self, text) -> int:
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

    def get_features(self, text) -> np.array:
        self.word_count = self.__word_count(text)
        self.char_count = self.__char_count(text)
        self.avg_word_count = self.__avg_word_count(text)
        self.stop_words_coun = self.__stop_words_coun(text)
        self.numbers_count = self.__numbers_count(text)
        self.title_word_count = self.__title_word_count(text)
        self.url_count = self.__url_count(text)
        self.bracket_mood_check = self.__bracket_mood_check(text)
        return np.array([self.word_count,
                         self.char_count,
                         self.avg_word_count,
                         self.stop_words_coun,
                         self.numbers_count,
                         self.title_word_count,
                         self.url_count,
                         self.bracket_mood_check,
                         ])

    def features_as_dataframe(self, text) -> pd.DataFrame:
        self.features_list = ['word_count',
                              'char_count',
                              'avg_word_count',
                              'stop_words_coun',
                              'numbers_count',
                              'title_word_count',
                              'url_count',
                              'bracket_mood_check',
                              ]
        self.features = self.get_features(text)
        return pd.DataFrame([self.features],
                            columns=self.features_list,
                            )