import argparse
import os
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AdamW
from dataset import EthicsDataset
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(train_loader, model, criterion, optimizer):
    model.train()
    
    total_loss = 0.0
    total_length = len(train_loader.dataset)
    
    with tqdm(total=len(train_loader), unit='step') as t:
        for batch1, batch2, labels in train_loader:
            inputs1 = {k: v.to(device).long() for k, v in batch1.items()}
            inputs2 = {k: v.to(device).long() for k, v in batch2.items()}
            
            optimizer.zero_grad()
            outputs1 = model(**inputs1)
            outputs2 = model(**inputs2)

            logits1 = outputs1.logits.squeeze()
            logits2 = outputs2.logits.squeeze()

            diffs = logits1 - logits2

            loss = criterion(diffs, labels.to(device).float())
            loss.backward()
            optimizer.step()

            total_loss += loss * len(labels)
            
            t.set_postfix(loss=f"{loss:.4f}")
            t.update(1)
    
    loss = total_loss / total_length
    
    print(f"Train Loss : {loss:.4f}")


def train(args):
    base_dir = './ethics'
    train_name = 'utilitarianism/util_train.csv'
    test_name = 'utilitarianism/util_test.csv'
    test_hard_name = 'utilitarianism/util_test_hard.csv'

    tokenizer = load_tokenizer(args)

    # load dataset
    train_dataset = EthicsDataset(tokenizer, os.path.join(base_dir, train_name), 'util')
    test_dataset = EthicsDataset(tokenizer, os.path.join(base_dir, test_name), 'util')
    test_hard_dataset = EthicsDataset(tokenizer, os.path.join(base_dir, test_hard_name), 'util')

    # load dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    test_hard_loader = DataLoader(test_hard_dataset, batch_size=args.batch_size, shuffle=False)

    model = load_model(args)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        print(f'< Epoch {epoch+1}/{args.epochs} >')
        
        # train
        train_epoch(train_loader, model, criterion, optimizer)
        
        # evaluate
        print('Test Dataset')
        test_acc = evaluate(model, test_loader)
        print('Test Hard Dataset')
        test_hard_acc = evaluate(model, test_hard_loader)

    return test_acc, test_hard_acc


@torch.no_grad()
def evaluate(model, test_loader):
    model.eval()
    cors = []

    for batch1, batch2, _ in test_loader:
        inputs1 = {k: v.to(device).long() for k, v in batch1.items()}
        inputs2 = {k: v.to(device).long() for k, v in batch2.items()}

        logits1 = model(**inputs1).logits
        logits2 = model(**inputs2).logits

        diffs = logits1 - logits2

        diffs = diffs.squeeze().detach().cpu().numpy()
        cors.append(diffs > 0)

    cors = np.concatenate(cors)
    acc = np.mean(cors)

    print(f'Accuracy {acc:.4f}')

    return acc
