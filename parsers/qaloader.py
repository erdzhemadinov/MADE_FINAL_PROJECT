#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 14:11:25 2020

@author: iavode
"""
import os
import sys
from logging import getLogger, INFO
from random import randint, seed
from time import time_ns

from pandas import DataFrame
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


from otvetmailparser import get_question_answers
from utils import log


PARSER_LOGGER = getLogger("otvet.mail.parser.logger")


def fill_questions_answes(question_answers, questions, answers):
    """Preprocess recevie question and answers and append to lists."""
    if question_answers:
        receive_question, receive_answers = question_answers
        if receive_answers:
            for answer in receive_answers:
                questions.append(receive_question)
                answers.append(answer)
            if not len(answers) % 1000:
                print(f"Get {len(answers)} answers.")
    return questions, answers


def questions_answers_loader(req_size=1800):
    """Get required quantity of answers."""
    seed(int(time_ns()))
    http = "https://otvet.mail.ru/question/"
    questions = []
    answers = []
    while len(answers) < req_size:
        index = randint(2, 222489031)
        url = http + str(index)
        log(PARSER_LOGGER, INFO, "url: %s" % url)
        question_answers = get_question_answers(url)
        questions, answers = fill_questions_answes(
            question_answers, questions, answers
        )
    return questions, answers


def _write_to_csv(questions, answers):
    """Write recieve questions and answers to csv file."""
    columns = ("question", "answer", "trollo")
    trollo = [float("nan") for _ in answers]
    data = [questions, answers, trollo]
    data = {
        columns[i]: data
        for i, data in enumerate(data)
    }
    data = DataFrame(data).sample(frac=1)
    data.to_csv("random_qa.csv", index=False)


def setup_parser(parser: ArgumentParser):
    """Create command line arguments."""
    parser.add_argument(
        "-raq", "--req-answer-quantity", dest="req_ans_quantity", type=int, required=True,
        default=None, help="Required quantity of url"
    )


def get_argparser():
    """Create argument parser."""
    description = (
        "Get required quantity of questions and answer from otvet.mail."
    )
    parser = ArgumentParser(
        prog="required questions and answer parser.", description=description,
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    return parser


def run():
    """Main function for run site parsing."""
    parser = get_argparser()
    setup_parser(parser)
    args = parser.parse_args()
    questions, answers = questions_answers_loader(args.req_ans_quantity)
    print(f"Get {len(answers)} answers.")
    _write_to_csv(questions, answers)


if __name__ == "__main__":
    run()