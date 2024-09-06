from .bert_modules.bert import BERT
import torch.nn as nn


class BERTModel:
    def __init__(self, args):
        self.bert = BERT(args)
        self.out = nn.Linear(self.bert.hidden, args.num_items + 1)

    def forward(self, x):
        x = self.bert(x)
        return self.out(x)
