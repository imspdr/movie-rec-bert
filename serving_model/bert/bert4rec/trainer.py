import torch
import torch.nn as nn
import torch.optim as optim

class Trainer:
    def __init__(self, model, train_loader):
        self.device = "cpu"
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=0.00)
        self.ce = nn.CrossEntropyLoss(ignore_index=0)

    def train(self, num_epoch):
        accum_iter = 0
        for epoch in range(num_epoch):
            accum_iter = self.train_one_epoch(accum_iter)
        torch.save(self.model, "bert4rec_model")

    def calculate_loss(self, batch):
        seqs, labels = batch
        logits = self.model(seqs)  # B x T x V
        logits = logits.view(-1, logits.size(-1))  # (B*T) x V
        labels = labels.view(-1)  # B*T
        loss = self.ce(logits, labels)
        return loss

    def train_one_epoch(self, accum_iter):
        print("one-epoch-start : " + str(accum_iter))
        self.model.train()
        whole_loss = 0
        for batch_idx, batch in enumerate(self.train_loader):
            batch_size = batch[0].size(0)
            batch = [x.to(self.device) for x in batch]

            self.optimizer.zero_grad()
            loss = self.calculate_loss(batch)
            loss.backward()
            whole_loss+=loss.item()
            self.optimizer.step()
            accum_iter += batch_size


        print("one-epoch-done : " + str(accum_iter) + " loss : " + str(whole_loss / len(self.train_loader)))
        return accum_iter


