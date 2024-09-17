from .bert_modules.bert import BERT
import torch.nn as nn


class BERTModel(nn.Module):
    def __init__(self, num_items, max_len=100):
        super().__init__()
        self.bert = BERT(num_items, max_len)
        self.out = nn.Linear(256, num_items + 1)

    def forward(self, x):
        x = self.bert(x)
        return self.out(x)
