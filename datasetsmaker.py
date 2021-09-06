#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 18:48:15 2020

@author: iavode
"""
from pandas import read_csv, concat
from numpy import nan

SAMPLE_SIZE = 100
NAMES = ("Djemadinov", "Zakladniy", "Ivanov", "Kovtun", "Obidin", "Odegov")


def get_grouped_dataset(question_answer):
    """Group dataset by answer."""
    group_by_target = question_answer.groupby(by="target")
    zero = group_by_target.get_group(0).sample(frac=1)
    one = group_by_target.get_group(1).sample(frac=1)
    two = group_by_target.get_group(2).sample(frac=1)
    return zero, one, two


def get_slice(dataset, idx):
    """Get silce of dataset from idx * SAMPLE to (idx + 1) * SAMPLE"""
    start, stop = idx * SAMPLE_SIZE, (idx + 1) * SAMPLE_SIZE
    return dataset.iloc[start: stop]


def process_row(row):
    """Concat row (sequence of dataset) and add new empty column."""
    new = concat(row).sample(frac=1)
    new["trollolo"] = nan
    return new


def write_datasets(question_answer, alls):
    """write created dataser and preprocess question_answer"""
    question_answer.sort_index(axis=0, inplace=True)
    question_answer.to_csv("./processs_question_answer.csv", index=False)
    for dataset, name  in zip(alls, NAMES):
        dataset.drop(columns=['target'],inplace=True)
        dataset.to_csv(f"./dataset_for_mark_{name}.csv", index=False)


def main():
    """Main function.

    Read, processings, output.

    """
    question_answer = read_csv(
        "./data_main_rl.csv", usecols=["target", "question", "answer"]
    )
    zero, one, two = get_grouped_dataset(question_answer)
    quantity = len(NAMES)
    zeros = [get_slice(zero, i) for i in range(quantity)]
    ones = [get_slice(one, i) for i in range(quantity)]
    twos = [get_slice(two, i) for i in range(quantity)]
    alls = [process_row(row) for row in zip(zeros, ones, twos)]
    # del row with index from created dataset from question_answer
    _ = [question_answer.drop(dataset.index, inplace=True) for dataset in alls]
    write_datasets(question_answer, alls)


if __name__ == "__main__":
    main()
