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
        for batch, labels in train_loader:
            inputs = {k: v.to(device).long() for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = model(**inputs)
            logits = outputs.logits.squeeze()

            loss = criterion(logits, labels.to(device).float())
            loss.backward()
            optimizer.step()

            total_loss += loss * len(labels)
            
            t.set_postfix(loss=f"{loss:.4f}")
            t.update(1)
    
    loss = total_loss / total_length
    
    print(f"Train Loss : {loss:.4f}")


def train(args):
    base_dir = './ethics'
    train_name = 'commonsense/cm_train.csv'
    test_name = 'commonsense/cm_test.csv'
    test_hard_name = 'commonsense/cm_test_hard.csv'

    tokenizer = load_tokenizer(args)

    # load dataset
    train_dataset = EthicsDataset(tokenizer, os.path.join(base_dir, train_name), 'cm')
    test_dataset = EthicsDataset(tokenizer, os.path.join(base_dir, test_name), 'cm')
    test_hard_dataset = EthicsDataset(tokenizer, os.path.join(base_dir, test_hard_name), 'cm')

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
    cor = 0
    total = 0

    for batch, labels in test_loader:
        inputs = {k: v.to(device).long() for k, v in batch.items()}

        logits = model(**inputs).logits

        output = logits.squeeze().detach().cpu().numpy()
        predictions = (output > 0).astype(int)

        labels = labels.detach().cpu().numpy()
        cor += (predictions == labels).sum()
        total += labels.shape[0]

    acc = cor / total
    print(f'Accuracy: {acc:.4f}')
    
    return acc
