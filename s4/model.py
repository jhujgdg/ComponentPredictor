import torch
import torch.nn as nn
from s4 import S4Block  # 确保你的 s4.py 中有这个类

class S4Net(nn.Module):
    def __init__(self, num_embeddings, num_classes, embedding_dim=128, d_model=128, dropout=0.1):
        super(S4Net, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.s4 = S4Block(d_model=d_model, dropout=dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x,_ = self.s4(x)
        x = x.permute(0, 2, 1)
        x_last = x[:, -1, :]
        return self.classifier(x_last)
