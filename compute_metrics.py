import csv
import os
import sys
from functools import reduce

import numpy as np
from sacrebleu.metrics import BLEU
import jsonlines
import glob
import re
from src.evaluation import rouge_score, f1_score, normalize_answer
from tqdm import tqdm



def calc_bleu(jsonl_path):
    jsonl_data = []

    with jsonlines.open(jsonl_path) as reader:
        for obj in reader:
            obj['answers'][0] = obj['answers'][0].replace('R:', '').strip()
            obj['generation'] = obj['generation'].replace('R:', '').strip()
            jsonl_data.append(obj)

    targets = [row['answers'][0] for row in jsonl_data]
    pred_text = [row['generation'] for row in jsonl_data]
    bleu = BLEU().corpus_score(pred_text, [targets])
    # print("===\npath:{}\n{}\n====\n".format(jsonl_path, bleu.format()))
    return bleu


def calc_rouge(jsonl_path):
    jsonl_data = []

    with jsonlines.open(jsonl_path) as reader:
        for obj in reader:
            obj['answers'][0] = obj['answers'][0].replace('R:', '').strip()
            obj['generation'] = obj['generation'].replace('R:', '').strip()
            jsonl_data.append(obj)

    targets = [row['answers'][0] for row in jsonl_data]
    pred_text = [row['generation'] for row in jsonl_data]
    rouge = [rouge_score(pred, [target]) for pred, target in zip(pred_text, targets)]
    rouge_mat = np.asmatrix(rouge)*100
    averaged = rouge_mat.mean(0).tolist()[0]
    return averaged


def calc_f1(jsonl_path):
    jsonl_data = []

    with jsonlines.open(jsonl_path) as reader:
        for obj in reader:
            obj['answers'][0] = obj['answers'][0].replace('R:', '').strip()
            obj['generation'] = obj['generation'].replace('R:', '').strip()
            jsonl_data.append(obj)

    targets = [row['answers'][0] for row in jsonl_data]
    pred_text = [row['generation'] for row in jsonl_data]
    score = [f1_score(pred, [target], normalize_answer) for pred, target in zip(pred_text, targets)]
    avg = np.asarray(score).mean()*100
    return avg


# # Accept file path as command line argument
# if len(sys.argv) > 1:
#     file_name = sys.argv[1]
# else:
#     file_name = 'ckpt/xl/ckpt/valid-resu

# file_name = "C:\\Users\\momen\\OneDrive\\Desktop\\MAI\\Thesis\\LAPDOG-main\\ckpt\\joint_grpo_v49\\valid-step-5000.jsonl"
file_name = "C:\\Users\\momen\\OneDrive\\Desktop\\MAI\\Thesis\\Models and Ckpt\\stage 2 - 20000 exp\\valid-step-20000.jsonl"

bleu = calc_bleu(file_name)
rouge = calc_rouge(file_name)
f1 = calc_f1(file_name)
print(f"""
File: {file_name}
BLEU: {bleu}
ROUGE: {rouge}
F1: {f1}
""")