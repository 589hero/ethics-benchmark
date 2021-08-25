import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    return tokenizer


def load_model(args):
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1)
    model = model.to(device)

    return model

