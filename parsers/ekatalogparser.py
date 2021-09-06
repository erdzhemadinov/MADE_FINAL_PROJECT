# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 06:14:09 2020.

@author: Odegov Ilya
"""
import os
import sys
from logging import getLogger, INFO
import re

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from bs4 import BeautifulSoup

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from utils import log, get_page


PARSER_LOGGER = getLogger("logger.e-katalog.parser")
PATTERN_HELPFUL = re.compile(r"'[0-9]{1,}'")
PATTERN_SCORE = re.compile(r"[0-9]{1}")


def _get_review_title_and_score(review):
    """Get review title and score."""
    title = review.find(name="div", class_="review-title")
    score = title.find(name="img").attrs.get("src", None)
    if score is None:
        return score, score
    score = int(re.findall(PATTERN_SCORE, score)[0])
    title = title.find("span")
    try:
        title = title.text
    except Exception:
        title = float("nan")
    return title, score


def _get_review_helpful(review):
    helpfuls = review.find(name="td", class_="review-helpful")
    helpfuls = re.findall(PATTERN_HELPFUL, helpfuls.text)
    helpfuls = [int(helpful.strip("'")) for i, helpful in enumerate(helpfuls)]
    return helpfuls


def _preproces_sequence(sequence):
    r"""Remove '-', \n, ' ' from input sequence."""
    try:
        return sequence.strip("-").strip("\n").strip(" ")
    except TypeError:
        return float("nan")


def _get_review_sign(sign_):
    """Get product plus or minus."""
    if not sign_:
        return float("nan")
    sign_ = sign_.contents[::2]
    sign_ = map(lambda seq: _preproces_sequence(seq) + ".", sign_)
    sign_ = " ".join(sign_)
    return sign_


def _get_review_param(review, class_):
    """Get review params in relation to recieved inputs, handle exceprion."""
    if class_ == "review-title":
        return _get_review_title_and_score(review)
    if class_ == "review-helpful":
        return _get_review_helpful(review)
    if class_ == "review-comment":
        review = review.find(name="div", class_=class_)
        if review:
            # .find(name="div", class_=class_).find("span")
            return review.find("span").text
        else:
            return float("nan")
    sign_ = review.find(name="div", class_=class_)
    return _get_review_sign(sign_)


def _parse_saparate_review(review):
    """Parse review, get review title, comment, plus, minus, score."""
    title, score = _get_review_param(review, class_="review-title")
    if not score:
        return None
    comment = _get_review_param(review, class_="review-comment")
    plus = _get_review_param(review, class_="review-plus")
    minus = _get_review_param(review, class_="review-minus")
    yes_sign, no_sign = _get_review_param(review, class_="review-helpful")
    return title, comment, plus, minus, yes_sign, no_sign, score


def _get_product_reviews(reviews):
    """Get all producte reviews params from page."""
    # titles, comments, pluses, minuses, helpfuls+, helpfuls-, scores
    results = [[], [], [], [], [], [], []]
    for review in reviews:
        result = _parse_saparate_review(review)
        if result:
            for element_list, element in zip(results, result):
                element_list.append(element)
    # titles, comments, pluses, minuses, helpfuls, scores
    return results


def get_product_reviews(product_url):
    """Get all reviews for recieved product url."""
    log(PARSER_LOGGER, INFO, "Get page.")
    page = get_page(product_url, PARSER_LOGGER)
    if not page:
        return None
    log(PARSER_LOGGER, INFO, "Get html content by BS.")
    html = BeautifulSoup(page.content, 'xml')
    product_reviews = html.find_all(name="table", class_="review-table")
    product_reviews = _get_product_reviews(product_reviews)
    return product_reviews


def _setup_parser(parser: ArgumentParser):
    """Create command line arguments."""
    parser.add_argument(
        "-u", "--url", dest="url", type=str, required=True, default=None,
        help="url for parsing."
    )


def _get_argparser():
    """Create argument parser."""
    description = "Parser e-katalog.ru product reviews."
    parser = ArgumentParser(
        prog="questions parser", description=description,
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    return parser


def run():
    """Main function for run site parsing."""
    parser = _get_argparser()
    _setup_parser(parser)
    args = parser.parse_args()
    log(PARSER_LOGGER, INFO, "Get url in args: %s" % args.url)
    product_reviews = get_product_reviews(args.url)
    print(*product_reviews, sep="\n")
#    _output(question_answers, url=args.url)


if __name__ == "__main__":
    run()
