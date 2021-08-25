import torch
import pandas as pd


class EthicsDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, csv_path, data, max_length=64):
        self.data = data
        if data == 'cm':
            df = pd.read_csv(csv_path)

            self.scenarios = df['input'].tolist()
            self.labels = df['label'].tolist()
            self.encodings = tokenizer(self.scenarios,
                                       max_length=max_length,
                                       padding='max_length',
                                       truncation=True)

        elif data == 'deontology':
            df = pd.read_csv(csv_path)

            self.scenarios = df['scenario'].tolist()
            self.execuses = df['excuse'].tolist()
            self.labels = df['label'].tolist()
            self.encodings = tokenizer(self.scenarios,
                                       self.execuses,
                                       max_length=max_length,
                                       padding='max_length',
                                       truncation=True)

        elif data == 'util':
            df = pd.read_csv(csv_path, header=None)

            self.sentence1 = df[0].tolist()
            self.sentence2 = df[1].tolist()
            self.encodings1 = tokenizer(self.sentence1,
                                        max_length=max_length,
                                        padding='max_length',
                                        truncation=True)
            self.encodings2 = tokenizer(self.sentence2,
                                        max_length=max_length,
                                        padding='max_length',
                                        truncation=True)
            self.labels = torch.ones(len(self.encodings1['input_ids']))

        else:
            df = pd.read_csv(csv_path)

            self.scenarios = df['scenario'].tolist()
            self.labels = df['label'].tolist()
            self.encodings = tokenizer(self.scenarios,
                                       max_length=max_length,
                                       padding='max_length',
                                       truncation=True)

        
    def __getitem__(self, idx):
        if self.data == 'util':
            item1 = {k: torch.Tensor(v[idx]) for k, v in self.encodings1.items()}
            item2 = {k: torch.Tensor(v[idx]) for k, v in self.encodings2.items()}
            labels = self.labels[idx]
            
            return item1, item2, labels
        else:
            item = {k: torch.Tensor(v[idx]) for k, v in self.encodings.items()}
            labels = self.labels[idx]
            
            return item, labels


    def __len__(self):
        return len(self.labels)
        