import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNNNet, self).__init__()
        self.input_size = input_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        input_x = F.one_hot(x.long(), num_classes=self.input_size).float()
        out, _ = self.rnn(input_x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

