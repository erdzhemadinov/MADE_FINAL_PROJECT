#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 15:46:28 2020

@author: Odegov Ilya
"""
import os
import torch
from torch import nn
from transformers.modeling_bert import BertConfig, BertModel


class Classifier(nn.Module):
    """Classifier class."""

    def __init__(
            self, embedder_or_name_or_path, path_to_load_model, out_features,
            device, dropout_prob=0.5):
        """Init class object."""
        nn.Module.__init__(self)
        if isinstance(embedder_or_name_or_path, BertModel):
            self._embedder = embedder_or_name_or_path
        elif path_to_load_model:
            config = BertConfig.from_json_file(
                os.path.join(path_to_load_model, "config.json")
            )
            embedder_or_name_or_path = os.path.join(
                path_to_load_model, embedder_or_name_or_path
            )
            self._embedder = BertModel(config)
            self._embedder.load_state_dict(
                torch.load(embedder_or_name_or_path, map_location=device)
            )
        else:
            self._embedder = BertModel.from_pretrained(
                embedder_or_name_or_path, output_hidden_states=True
            )
        for param in self._embedder.parameters():
            param.requires_grad = True
        size = self._embedder.config.hidden_size
        self._classifier = nn.Linear(
            in_features=size, out_features=out_features)
        self._dropout = nn.Dropout(p=dropout_prob)
        self._activation = nn.ReLU()
        self._batch_norm = nn.BatchNorm1d(num_features=size)
        self._init_classifier()

    def _init_classifier(self):
        """Init weights of classifier layer."""
        nn.init.xavier_uniform(self._classifier.weight)
        nn.init.uniform(self._classifier.bias, -0.01, 0.01)

    def forward(self, inpt):
        """Model forward step."""
        # input tokens, mask and token type ids for embedder input
        inpt_ids, inpt_mask = (
            inpt["input_ids"], inpt["attention_mask"]
        )
        shape = inpt_ids.shape
        embedding = self._embedder(
            input_ids=inpt_ids, attention_mask=inpt_mask
        )
        # get only cls tokken. Need to think about different variant.
        # size: batch_size x 768
        output = embedding[0][:, 0, :]
        assert output.shape[0] == shape[0], "Wrong shape."
        output = self._dropout(self._batch_norm(self._activation(output)))
        output = self._classifier(output)
        return output

    @property
    def embedder(self):
        """Getter for embedder."""
        return self._embedder

    @property
    def classifier(self):
        """Getter for classifier."""
        return self._classifier
