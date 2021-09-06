#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 15:13:47 2020

@author: Odegov Ilya
"""
import os
import sys
from copy import deepcopy
from json import loads, dumps
from logging import getLogger, INFO


from numpy import asarray
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
import torch
from torch.nn import BCEWithLogitsLoss
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from utils import log


class BertTainer:
    """Class implement train and save bert model if metric grow."""

    def __init__(
            self, model, optimizer, sheduler, loss, train_dataloader,
            validte_dataloader, save_model_path, model_sate_dict_file_name):
        """Init class object."""
        self._model = model
        self._optimizer = optimizer
        self._sheduler = sheduler
        self._loss = loss
        self._train_dataloader = train_dataloader
        self._validte_dataloader = validte_dataloader
        self._device = next(model.parameters()).device
        self._save_model_path = save_model_path
        self._model_sate_dict_file_path = os.path.join(
            self._save_model_path, model_sate_dict_file_name
        )
        self._writer = SummaryWriter()
        self._logger = getLogger("logger.trainer")

    def train(self, epochs, test_model_class=None):
        """Start main training loop."""
        if test_model_class:
            try:
                with open("best_test_metric.json", "r") as fin:
                    best_test_metrics = loads(fin.read())
            except FileNotFoundError:
                best_test_metrics = [0.6]
            best_test_metric = best_test_metrics[-1]
        else:
            best_validate_loss = float("inf")
        self._i_train, self._i_validate = 0, 0
        for epoch in range(epochs):
            print(f"Start {epoch + 1 } / {epochs} epoch.")
            log(self._logger, INFO, f"Start {epoch + 1 } / {epochs} epoch.")
            # train
            train_loss = self._train_loop()
            log(
                self._logger, INFO,
                f"Train loop is finished. Loss: {round(train_loss, 5)}"
            )
            self._writer.add_scalar("Train epochs loss", train_loss, epoch + 1)
            # validate
            valid_loss = self._validate_loop()
            log(
                self._logger, INFO,
                f"Validate loop is finished. Loss: {round(valid_loss, 5)}"
            )
            self._writer.add_scalar("Validate epochs loss", valid_loss, epoch + 1)
            self._sheduler.step(valid_loss)
            # save model and best metric / loss
            if test_model_class:
                metric = self._test_loop(test_model_class)
                if metric > best_test_metric:
                    best_test_metric = metric
                    # save model
                    self._save_model()
                    best_test_metrics.append(metric)
                    log(
                        self._logger, INFO,
                        f"New best metric. ROC_AUC: {round(metric, 5)}"
                    )
                with open("best_test_metric.json", "w") as fout:
                    fout.write(dumps(best_test_metrics))
            else:
                if valid_loss < best_validate_loss:
                    best_validate_loss = valid_loss
                    # save model and config
                    self._save_model()

    def _step(self, inpt, target):
        """Compute common part of train and validate loop."""
        self._optimizer.zero_grad()
        # question tokens, mask and token type ids for model input
        for key, value in inpt.items():
            inpt[key] = value.to(self._device)
        output = self._model(inpt)
        codition = all((
            output.shape[0] == target.shape[0],
            output.shape[1] == self._model.classifier.out_features
        ))
        assert codition, "Wrong shapes: output, target."
        if isinstance(self._loss, BCEWithLogitsLoss):
            output = output.view(-1)
            target = target.type(torch.float)
        loss = self._loss(output.cpu(), target)
        return loss

    def _train_loop(self):
        """One train loop."""
        self._model.train()
        epoch_loss = 0
        loader = tqdm(
            self._train_dataloader, total=len(self._train_dataloader)
        )
        i = 0
        for inpt, target in loader:
            self._optimizer.zero_grad()
            loss = self._step(inpt, target)
            loss.backward()
            clip_grad_norm_(self._model.parameters(), 10.0)
            self._optimizer.step()
            epoch_loss += loss.item()
            loader.set_description(
                f"Current mean loss: {round(epoch_loss / (i + 1), 4)}"
            )
            if not (i % 50):
                self._writer.add_scalar(
                    "Train loss", round(epoch_loss / (i + 1), 4), self._i_train
                )
            i += 1
            self._i_train += 1
        return epoch_loss / len(self._train_dataloader)

    @torch.no_grad()
    def _validate_loop(self):
        """One validate loop."""
        self._model.eval()
        epoch_loss = 0
        loader = tqdm(
            self._validte_dataloader, total=len(self._validte_dataloader)
        )
        i = 0
        for inpt, target in loader:
            loss = self._step(inpt, target)
            epoch_loss += loss.item()
            loader.set_description(
                f"Current mean loss: {round(epoch_loss / (i + 1), 4)}"
            )
            if not (i % 5):
                self._writer.add_scalar(
                    "Validate loss", round(epoch_loss / (i + 1), 4),
                    self._i_validate
                )
            i += 1
            self._i_validate += 1
        return epoch_loss / len(self._validte_dataloader)

    def _test_loop(self, test_model_class):
        """Train and test full model with fine-tuning bert in the end of each epoch."""
        test_model = test_model_class(self._model.embedder)
        roc_auc = test_model.fit()
        return roc_auc

    def _save_model(self):
        """Save model and config file."""
        torch.save(
            self._model.embedder.state_dict(),
            self._model_sate_dict_file_path,
            )
        self._model.embedder.config.save_pretrained(self._save_model_path)

    @property
    def model(self):
        """Getter for model."""
        return self._model

    @property
    def sheduler(self):
        """Getter for sheduler."""
        return self._sheduler

    @property
    def optimizer(self):
        """Getter for optimizer."""
        return self._optimizer

    @property
    def device(self):
        """Getter for using device."""
        return self._device

    @property
    def save_path(self):
        """Getter path where save model."""
        return self._save_path
