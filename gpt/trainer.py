#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 23:14:57 2020

@author: iavode
"""
from tqdm import tqdm
import torch


class Trainer:
    """Class for train recive model."""

    def __init__(
            self, model, optimizer, sheduler, loss,
            train_dataloader, test_dataloader):
        """Init class obj."""
        self._model = model
        self._optimizer = optimizer
        self._sheduler = sheduler
        self._loss = loss
        self._train_dataloader = train_dataloader
        self._test_dataloader = test_dataloader
        self._device = next(model.parameters()).device

    def train(self, epochs):
        """Start main learning loop."""
        train_history, valid_history = [], []
        best_valid_loss = float('inf')
        for epoch in range(epochs):
            print(f"Start {epoch + 1 } epoch.")
            train_loss = self._train_loop()
            valid_loss = self._validate_loop()
            self._sheduler.step(valid_loss)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self._model.state_dict(), "sber_gpt3.pt")
            train_history.append(train_loss)
            valid_history.append(valid_loss)

    def _train_loop(self):
        """One train loop."""
        self._model.train()
        epoch_loss = 0
        history = []
        for questions, answers in tqdm(self._train_dataloader):
            self._optimizer.zero_grad()
            loss = self._step(questions, answers)
            loss.backward()
            self._optimizer.step()
            epoch_loss += loss.item()
            history.append(loss.cpu().data.numpy())
        return epoch_loss / len(self._train_dataloader)

    @torch.no_grad()
    def _validate_loop(self):
        """One validate loop."""
        self._model.eval()
        epoch_loss = 0
        history = []
        for questions, answers in tqdm(self._test_dataloader):
            loss = self._step(questions, answers)
            epoch_loss += loss.item()
            history.append(loss.cpu().data.numpy())
        return epoch_loss / len(self._test_dataloader)

    def _step(self, questions, answers):
        """Compute loss for train and validate loop iteration."""
        q_ids, q_mask, q_token_type_ids = (
            questions["ids"].to(self._device),
            questions["mask"].to(self._device),
            questions["token_type_ids"].to(self._device)
        )
        output = self._model(
            input_ids=q_ids, attention_mask=q_mask,
            token_type_ids=q_token_type_ids
        )
        shape = output[0].shape
        output = output[0].view(-1, shape[-1])
        shape = answers["ids"].shape
        target = answers["ids"].view(-1).to(self._device)
        loss = self._loss(output, target)
        return loss

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
