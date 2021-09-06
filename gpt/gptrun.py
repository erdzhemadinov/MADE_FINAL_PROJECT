#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 20:59:18 2020

@author: Odegov Ilya
"""
import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from time import time

from transformers import pipeline

from utils import (
    set_seed, create_tokenizer, create_model
)


def _create_pipiline(tokenizer, model, device, framework):
    """Create text generation pipiline."""
    tg_params = dict(
        task="text-generation", tokenizer=tokenizer, model=model,
        framework="pt", device=device,
    )
    text_generation_pipiline = pipeline(**tg_params)
    return text_generation_pipiline


class TextGenerator:
    """Text generator pipiline."""

    def __init__(self, tokenizer, model, device=-1, framework="pt"):
        """Init class object."""
        set_seed(int(time()))
        tokenizer = create_tokenizer(tokenizer)
        model = create_model(model)
        self._text_generation_pipiline = _create_pipiline(
            tokenizer, model, device, framework)

    def __call__(self, seqs):
        """Call class object."""
        seqs = [seqs] if isinstance(seqs, str) else seqs
        max_length = max(map(len, seqs)) * 2
        return self._text_generation_pipiline(seqs, max_length=max_length)


def create_generator(tokenizer, model, framework="pt", device=-1):
    """Create text generator."""
    tg_params = dict(
        tokenizer=tokenizer, model=model,
        framework=framework, device=device,
    )
    text_generator = TextGenerator(**tg_params)
    return text_generator


def run():
    """Start script."""
    gpt = "sberbank-ai/rugpt3large_based_on_gpt2"
    generator = create_generator(gpt, gpt)
    print(generator("")[0]["generated_text"])


if __name__ == "__main__":
     run()
