#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 22:56:13 2020

@author: iavode
"""
import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from torch import Tensor, long
from torch.utils.data import Dataset


class QADataset(Dataset):
    """Class for preprocess auestion-answer dataset."""

    def __init__(self, data, tokenizer, q_max_len, a_max_len):
        """Object class init."""
        self._tokenizer = tokenizer
        self._data = data
        self._q_max_len = q_max_len
        self._a_max_len = a_max_len

    def __len__(self,):
        """Get len of object."""
        return len(self._data.question)

    def _tokenize(self, sequence, max_length):
        """Tokenize sequence."""
        sequence = self._tokenizer(
            sequence, None, truncation=True, max_length=max_length,
            return_token_type_ids=True, padding="max_length",
        )
        return sequence

    def _process_sequence(self, sequence, max_length):
        """Tokenize sequence, process sequence and create output dict."""
        sequence = self._tokenize(sequence, max_length)
        ids = Tensor(sequence["input_ids"]).to(long)
        mask = Tensor(sequence["attention_mask"]).to(long)
        token_type_ids = Tensor(sequence["token_type_ids"]).to(long)
        sequence = dict(
            ids=ids, mask=mask, token_type_ids=token_type_ids
        )
        return sequence

    def __getitem__(self, index):
        """Get preprocess item by index."""
        question = self._data["question"].iloc[index]
        answer = self._data["answer"].iloc[index]
        question = self._process_sequence(question, self._q_max_len)
        answer = self._process_sequence(answer, self._a_max_len)
        return question, answer
