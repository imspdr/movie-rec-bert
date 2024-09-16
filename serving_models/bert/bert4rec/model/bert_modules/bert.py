from torch import nn as nn
from ..bert_modules.embedding import BERTEmbedding
from ..bert_modules.transformer import TransformerBlock

class BERT(nn.Module):
    def __init__(self, num_items):
        super().__init__()

        # self.init_weights()
        max_len = 20
        n_layers = 2
        heads = 4
        vocab_size = num_items + 2
        hidden = 256
        dropout = 0.1

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden, max_len=max_len, dropout=dropout)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x

    def init_weights(self):
        pass
