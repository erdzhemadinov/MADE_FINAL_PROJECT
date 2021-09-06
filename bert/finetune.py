#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 09:34:04 2020

@author: Odegov Ilya
"""
import os
import sys
from logging import getLogger, INFO
from time import sleep, time

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pandas import read_csv
from tqdm import tqdm
import torch
from torch import cuda
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers.tokenization_bert import BertTokenizer

from classifier import Classifier
from modeltrainer import BertTainer

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(CURRENT_DIR)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from models.bertolet import IsTrollClassifierModel
from utils import DataLoader
from utils import smart_batching, log, set_seed, cleanup


def _create_model_and_loss(args, number_of_class, device):
    """Create maodel, loss in relation to parsing args and input dataset."""
    bert = args.load_model_name
    path = args.load_model_path
    loss, model = ((
        BCEWithLogitsLoss(),
        Classifier(bert, path, number_of_class - 1, device,)
        )
        if number_of_class == 2 else (
            CrossEntropyLoss(),
            Classifier(bert, path, number_of_class, device)
        )
    )
    return model.to(device), loss


def finetune(args):
    """Set up and sturt fine-tune operation."""
    tqdm.pandas()
    os.makedirs(args.save_model_path, exist_ok=True)
    logger = getLogger("logger.finetune")
    log(logger, INFO, "Start training with params:")
    for arg, value in sorted(vars(args).items()):
        log(logger, INFO, "Argument %s: %r" % (arg, value))
    # Start fine-tuning process.
    set_seed(int(time()))
    print(f"Read {args.path_train_dataset} dataset.")
    log(logger, INFO, f"Read {args.path_train_dataset} dataset.")
    dataset = read_csv(
        os.path.join(CURRENT_DIR, args.path_train_dataset),
        usecols=["text", "target"]
    ).iloc[:]
    log(logger, INFO, "")
    number_of_class = dataset["target"].unique().size
    print("Claean dataset.")
    log(logger, INFO, "Claean dataset.")
    dataset["text"] = dataset["text"].progress_apply(cleanup)
    print("Load tokenizer.")
    log(logger, INFO, "Load tokenizer.")
    tokenizer = BertTokenizer.from_pretrained(
       args.load_tokenizer_name, do_lower_case=True
    )
    num_batches = len(dataset)
    border = int(num_batches * args.split_ration)
    dataset = dataset.sample(frac=1)
    train_dataset, validte_dataset = dataset[:border], dataset[border:]
    # dataset == [batch, batch, batch, ..., batch]
    print("Start smart batching.")
    log(logger, INFO, "Start smart batching.")
    print("Train dataset.")
    log(logger, INFO, "Train dataset.")
    sleep(0.25)
    train_dataset = smart_batching(
        train_dataset, tokenizer, max_length=args.max_length,
        batch_size=args.batch_size,
    )
    log(logger, INFO, "Batch quantity: %s." % len(train_dataset))
    print("Validate dataset")
    log(logger, INFO, "Validate dataset.")
    sleep(0.25)
    validte_dataset = smart_batching(
        validte_dataset, tokenizer, max_length=args.max_length,
        batch_size=args.batch_size * 16,
    )
    log(logger, INFO, "Batch quantity: %s." % len(validte_dataset))
    print("Create loaders.")
    log(logger, INFO, "Create loaders.")
    train_dataloader = DataLoader(
        train_dataset, shuffle_=True, drop_last=True
    )
    validte_dataloader = DataLoader(
        validte_dataset, shuffle_=False, drop_last=True
    )
    device = torch.device("cuda:0" if cuda.is_available() else "cpu")
    print("Create models.")
    log(logger, INFO, "Create models.")
    model, loss = _create_model_and_loss(args, number_of_class, device)
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    lr_scheduler = ReduceLROnPlateau(optimizer, patience=1, factor=0.1)
    print("Create trainer.")
    log(logger, INFO, "Create trainer.")
    save_model_path, save_model_name = (
        args.save_model_path, args.save_model_name
    )
    trainer_params = dict(
        model=model, optimizer=optimizer, sheduler=lr_scheduler, loss=loss,
        train_dataloader=train_dataloader, validte_dataloader=validte_dataloader,
        save_model_path=save_model_path, model_sate_dict_file_name=save_model_name
    )
    trainer = BertTainer(**trainer_params)
    print("Run train.")
    trainer.train(epochs=args.epochs, test_model_class=IsTrollClassifierModel)


def _get_argparser():
    """Create argument parser."""
    description = "Fine-tune BERT-like model.."
    parser = ArgumentParser(
        prog="You fine-tune and you fine-tune.", description=description,
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    return parser


def _setup_parser(parser):
    """Create command line arguments."""
    # load and save path
    # load section
    # model load section
    parser.add_argument(
        "-lmn", "--load-model-name", dest="load_model_name", type=str,
        required=False, default="DeepPavlov/rubert-base-cased-conversational",
        help=(
            "If in --load-model-path is path to folder with model. " +
            "It mist be a model name. Else it is the model name from huggingface.co"
        )
    )
    parser.add_argument(
        "-lmp", "--load-model-path", dest="load_model_path", type=str,
        required=False, default=None,
        help="Path to folder with bert model."
    )
    # tokenizer load section
    parser.add_argument(
        "-ltn", "--load-tokenizer-name", dest="load_tokenizer_name", type=str,
        required=False, default="DeepPavlov/rubert-base-cased-conversational",
        help="Tokenizer name on huggingface.co"
    )
    # save section
    # model save section
    parser.add_argument(
        "-smp", "--save-model-path", dest="save_model_path", type=str,
        required=False, default="bert_fine_tune", help=(
            "Path to folder where will be save bert model and config file."
        )
    )
    parser.add_argument(
        "-smn", "--save-model-name", dest="save_model_name", type=str,
        required=False, default="bert_ft.pt", help=(
            "Name for save bert fine-tune model."
        )
    )
    # dataset arguments
    parser.add_argument(
        "-ptrd", "--path-to-train-dataset", dest="path_train_dataset",
        type=str, required=True, default=None, help="Path to train dataset."
    )
    # arguments for otimizer, split data and etc.
    parser.add_argument(
        "-e", "--epochs", dest="epochs", type=int,
        required=False, default=2, help="Fine-tune epochs."
    )
    parser.add_argument(
        "-lr", "--lr", dest="learning_rate", type=float,
        required=False, default=3e-5, help="Optimizer learning rate."
    )
    parser.add_argument(
        "-wd", "--weight-decay", dest="weight_decay", type=float,
        required=False, default=0.001, help="Optimizer weight decay."
    )
    parser.add_argument(
        "-ml", "--max-length", dest="max_length", type=int,
        required=False, default=512, help="Max tokens quantity."
    )
    parser.add_argument(
        "-bs", "--batch-size", dest="batch_size", type=int,
        required=False, default=4, help="Size of the the train batch."
    )
    parser.add_argument(
        "-sr", "--split-ratio", dest="split_ration", type=float,
        required=False, default=0.85, help="Train/validate plit ration."
    )
    parser.add_argument(
        "-ubhs", "--use-bert-hidden-state", dest="use-bert-hidden-state",
        type=str, required=False, default="cls",
        help=(
            "Define how to use bert output. Use only cls token by default. " +
            "Aviable format: 1) start-stop-step, python slice-like; " +
            "2) 1,3,5,7,9 - user-define index squence separeate by common."
        )
    )


def run():
    """
    Start programm.

    Entrance to fine-tune. Create argparser. Setup it. Get args and start.
    """
    parser = _get_argparser()
    _setup_parser(parser)
    args = parser.parse_args()
    finetune(args)


if __name__ == "__main__":
    run()
    pass
