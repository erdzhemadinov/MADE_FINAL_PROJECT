#!/bin/bash
python finetune.py -lmn bert -smn bert_ft.pt -smp bert_fine_tune -ptrd datasets/process_text.csv -lr 3e-4 -wd 0.01 -bs 4
# python finetune.py -lmn "DeepPavlov/rubert-base-cased-conversational" -smn bert_ft.pt -smp bert_fine_tune -ptrd datasets/process_text.csv -lr 3e-4 -wd 0.01 -bs 4
sleep 1s
python finetune.py -lmn bert_ft.pt -lmp bert_fine_tune -smn bert_ft.pt -smp bert_fine_tune -ptrd datasets/process_neg_pos_tweet.csv -lr 3e-4 -wd 0.01 -bs 32 -e 1
sleep 1s
python finetune.py -lmn bert_ft.pt -lmp bert_fine_tune -smn bert_ft.pt -smp bert_fine_tune -ptrd datasets/process_kaggle.csv -lr 3e-4 -wd 0.01 -bs 4
sleep 1s
python finetune.py -lmn bert_ft.pt -lmp bert_fine_tune -smn bert_ft.pt -smp bert_fine_tune -ptrd datasets/process_joke.csv -lr 3e-4 -wd 0.01 -bs 16 -e 1
sleep 5s
python finetune.py -lmn bert_ft.pt -lmp bert_fine_tune -smn bert_ft.pt -smp bert_fine_tune -ptrd datasets/process_threat.csv -lr 3e-4 -wd 0.01 -bs 4 -e 1
sleep 1s
python finetune.py -lmn bert_ft.pt -lmp bert_fine_tune -smn bert_ft.pt -smp bert_fine_tune -ptrd datasets/process_big_toxic_balance.csv -lr 3e-4 -wd 0.01 -bs 4
sleep 1s
python finetune.py -lmn bert_ft.pt -lmp bert_fine_tune -smn bert_ft.pt -smp bert_fine_tune -ptrd datasets/process_qa.csv -lr 3e-4 -wd 0.01 -bs 4
sleep 1s
python finetune.py -lmn bert_ft.pt -lmp bert_fine_tune -smn bert_ft.pt -smp bert_fine_tune -ptrd datasets/process_comment_summary.csv -lr 3e-4 -wd 0.01 -bs 4
sleep 1s
python finetune.py -lmn bert_ft.pt -lmp bert_fine_tune -smn bert_ft.pt -smp bert_fine_tune -ptrd datasets/process_best.csv -lr 3e-4 -wd 0.01 -bs 4 -sr 0.4
sleep 1s
python finetune.py -lmn bert_ft.pt -lmp bert_fine_tune -smn bert_ft.pt -smp bert_fine_tune -ptrd datasets/process_tkk_tweet.csv -lr 3e-4 -wd 0.01 -bs 32 -e 3
sleep 1s
python finetune.py -lmn bert_ft.pt -lmp bert_fine_tune -smn bert_ft.pt -smp bert_fine_tune -ptrd datasets/process_bank_tweet.csv -lr 3e-4 -wd 0.01 -bs 32 -e 3
sleep 1s
python finetune.py -lmn bert_ft.pt -lmp bert_fine_tune -smn bert_ft.pt -smp bert_fine_tune -ptrd datasets/process_spam_best.csv -lr 3e-4 -wd 0.01 -bs 4
