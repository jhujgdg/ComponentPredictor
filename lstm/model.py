import torch.nn as nn

class LSTMNet(nn.Module):
    def __init__(self, num_classes, num_embeddings, embedding_dim=64, hidden_size=128, num_layers=1,
                 batch_first=True, bidirectional=False, dropout=0.1):
        super(LSTMNet, self).__init__()
        self.num_classes = num_classes
        self.num_embeddings = num_embeddings

        if self.num_embeddings > 0:
            self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=batch_first,
                            bidirectional=bidirectional, dropout=dropout)

        self.classify = nn.Linear(hidden_size * num_directions, self.num_classes)

    def forward(self, x):
        if self.num_embeddings > 0:
            x = self.embedding(x)

        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.classify(x)
        return x





