import torch, torch.nn as nn
from .tokens import VOCAB
class MusicLSTM(nn.Module):
    """Genre-conditioned 2-layer LSTM predicting the next 16th-note token."""
    def __init__(self, n_genres, emb=64, gemb=16, hidden=256):
        super().__init__()
        self.emb, self.gemb = nn.Embedding(VOCAB, emb), nn.Embedding(n_genres, gemb)
        self.lstm = nn.LSTM(emb+gemb, hidden, 2, batch_first=True, dropout=0.2)
        self.out = nn.Linear(hidden, VOCAB)
        self.cfg = dict(n_genres=n_genres, emb=emb, gemb=gemb, hidden=hidden)
    def forward(self, x, g, h=None):
        ge = self.gemb(g).unsqueeze(1).expand(-1, x.size(1), -1)
        y, h = self.lstm(torch.cat([self.emb(x), ge], -1), h)
        return self.out(y), h
