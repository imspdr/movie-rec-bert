import random
import torch
import torch.utils.data as data_utils

class BertTrainDataset(data_utils.Dataset):
    def __init__(self, u2seq, n_items, max_len=100, mask_prob=0.15):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = n_items + 1
        self.num_items = n_items
        self.rng = random.Random(6541)

    def __len__(self):
        return len(self.users * 2)

    def __getitem__(self, index):
        user = self.users[index // 2]
        seq = self.u2seq[user]

        tokens = []
        labels = []

        if index % 2 == 0:
            for s in seq:
                prob = self.rng.random()
                if prob < self.mask_prob:
                    prob /= self.mask_prob
                    tokens.append(self.mask_token)
                    labels.append(s)
                else:
                    tokens.append(s)
                    labels.append(0)

        else:
            for i, s in enumerate(seq):
                if i == len(seq) - 1:
                    tokens.append(self.mask_token)
                    labels.append(s)
                else:
                    tokens.append(s)
                    labels.append(0)

        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]
        mask_len = self.max_len - len(tokens)

        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels

        return torch.LongTensor(tokens), torch.LongTensor(labels)

class BertEvalDataset(data_utils.Dataset):
    def __init__(self, seq, max_len=100):
        self.input_seq = seq
        self.max_len = max_len

    def __len__(self):
        return len(self.input_seq)

    def __getitem__(self, index):
        seq = self.input_seq[index]

        tokens = []

        for s in seq:
            tokens.append(s)

        tokens = tokens[-self.max_len:]
        mask_len = self.max_len - len(tokens)

        tokens = [0] * mask_len + tokens

        return [torch.LongTensor(tokens)]

