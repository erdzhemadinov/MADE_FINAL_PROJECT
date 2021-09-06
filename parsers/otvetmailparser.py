# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 11:56:15 2020

@author: Odegov Ilya
"""
import os
import sys

from logging import getLogger, DEBUG, INFO
from json import loads

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from bs4 import BeautifulSoup


current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils import log, get_page


PARSER_LOGGER = getLogger("logger.otvet.mail.parser")


def _parser_question(question_answers):
    """Get question text if text does not exist get question title."""
    question = question_answers.get("text", None)
    title = question_answers.get("name", None)
    if not question and not title:
        return None
    if not question:
        return title
    if not title:
        return question
    return "\n".join((title, question))


def _parser_suggest_answers(question_answers):
    """Get list of answers."""
    answers_text = []
    suggest_answers = question_answers.get("suggestedAnswer", None)
    if not suggest_answers:
        log(PARSER_LOGGER, INFO, "Suggested answer is empty.")
        return answers_text
    for answer in suggest_answers:
        answer = answer.get("text", "")
        if not answer:
            continue
        log(PARSER_LOGGER, DEBUG, "Get: %s" % answer)
        answers_text.append(answer)
    return answers_text


def get_question_answers(url):
    """Get question and answers from input url."""
    log(PARSER_LOGGER, INFO, "Get page.")
    page = get_page(url, PARSER_LOGGER)
    if not page:
        return None
    log(PARSER_LOGGER, INFO, "Get html content by BS.")
    html = BeautifulSoup(page.content, 'xml')
    log(PARSER_LOGGER, INFO, "Get question answers block")
    text = html.find_all(name="script")[0].text
    question_answers = loads(text)["@graph"][0].get("mainEntity", None)
    if question_answers:
        log(PARSER_LOGGER, INFO, "Get question.")
        question = _parser_question(question_answers)
        if question:
            log(PARSER_LOGGER, INFO, "Getting question is done.")
            log(PARSER_LOGGER, INFO, "Start get answers from url.")
            log(PARSER_LOGGER, DEBUG, question)
            answers = _parser_suggest_answers(question_answers)
            log(PARSER_LOGGER, INFO, "Getting answers is done.")
            return question, answers
    return None


def _output(result, url):
    """Output recived results."""
    if result:
        if result[1]:
            _print_results(*result)
            return
        msg = f"For {url} is no answers"
    else:
        msg = "Question or url is incorrect format."
    log(PARSER_LOGGER, INFO, msg)
    print(msg)


def _print_results(questions, answers):
    """Print parsing result on display."""
    print(f"Questions:\n{questions}")
    print("Answers:")
    for i, answer in enumerate(answers, 1):
        print(f"{i}) {answer}")


def _setup_parser(parser: ArgumentParser):
    """Create command line arguments."""
    parser.add_argument(
        "-u", "--url", dest="url", type=str, required=True, default=None,
        help="url for parsing"
    )


def _get_argparser():
    """Create argument parser."""
    description = "Parser otvet.mail questions."
    parser = ArgumentParser(
        prog="questions parser", description=description,
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    return parser


def run():
    """Run site parsing."""
    parser = _get_argparser()
    _setup_parser(parser)
    args = parser.parse_args()
    log(PARSER_LOGGER, INFO, "Get url in args: %s" % args.url)
    question_answers = get_question_answers(args.url)
    _output(question_answers, url=args.url)


if __name__ == "__main__":
    run()
