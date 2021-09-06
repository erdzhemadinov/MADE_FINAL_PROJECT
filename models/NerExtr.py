# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 19:02:36 2020

@author: User
"""

import numpy as np
import pandas as pd
import natasha

from natasha import NamesExtractor
from natasha import AddrExtractor
from natasha import MoneyExtractor
from natasha import DatesExtractor
from natasha import MorphVocab


class NerExtractor():

    morph = MorphVocab()

    def __get_count_names(self, text, morph=morph) -> int:
        lst = []
        extractor = NamesExtractor(morph)
        for match in extractor(text):
            lst.append(match.fact)
        return len(lst)
    
    def __get_count_addr(self, text, morph=morph) -> int:
        lst = []
        extractor = AddrExtractor(morph)
        for match in extractor(text):
            lst.append(match.fact)
        return len(lst)
    
    def __get_count_date(self, text, morph=morph) -> int:
        lst = []
        extractor = MoneyExtractor(morph)
        for match in extractor(text):
            lst.append(match.fact)
        return len(lst)
    
    def __get_count_money(self, text, morph=morph) -> int:
        lst = []
        extractor = DatesExtractor(morph)
        for match in extractor(text):
            lst.append(match.fact)
        return len(lst)
    
    def get_features(self, text) -> np.array:
        self.count_names = self.__get_count_names(text)
        self.count_addr = self.__get_count_addr(text)
        self.count_date = self.__get_count_date(text)
        self.count_money = self.__get_count_money(text)
        return np.array([self.count_names,
                         self.count_addr,
                         self.count_date,
                         self.count_money,
                         ])