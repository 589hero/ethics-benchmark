import argparse
import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from transformers.trainer_utils import set_seed
from finetune import (
    justice_finetune,
    virtue_finetune,
    deontology_finetune,
    util_finetune,
    commonsense_finetune
)


def main(args):
    set_seed(args.seed)

    # justice
    justice_acc, justice_em, justice_hard_acc, justice_hard_em = justice_finetune.train(args)

    # virtue ethics
    virtue_acc, virtue_em, virtue_hard_acc, virtue_hard_em = virtue_finetune.train(args)

    # deontology
    deontology_acc, deontology_em, deontology_hard_acc, deontology_hard_em = deontology_finetune.train(args)

    # utilitarianism
    util_acc, util_hard_acc = util_finetune.train(args)

    # commonsense ethics
    args.max_length = 512
    cm_acc, cm_hard_acc = commonsense_finetune.train(args)


    ethics_results = {
        'justiceAcc': justice_acc,
        'justiceEM': justice_em,
        'justiceHardAcc': justice_hard_acc,
        'justiceHardEM': justice_hard_em,
        'virtueAcc': virtue_acc,
        'virtueEM': virtue_em,
        'virtueHardAcc': virtue_hard_acc,
        'virtueHardEm': virtue_hard_em,
        'deontologyAcc': deontology_acc,
        'deontologyEm': deontology_em,
        'deontologyHardAcc': deontology_hard_acc,
        'deontologyHardEm': deontology_hard_em,
        'utilAcc': util_acc,
        'utilHardAcc': util_hard_acc,
        'commonsenseAcc': cm_acc,
        'commonsenseHardAcc': cm_hard_acc
    }

    with open("ethics_results.json", "w") as f:
        json.dump(ethics_results, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    main(args)