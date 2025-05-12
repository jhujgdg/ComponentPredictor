import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=0, dropout=0.0):
        super(TemporalBlock, self).__init__()
        self.ll_conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                  dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.dropout1 = nn.Dropout(dropout)

        self.ll_conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                  dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.dropout2 = nn.Dropout(dropout)

        self.skip_connection = nn.Conv1d(in_channels, out_channels,
                                         kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        res = x
        x = self.ll_conv1(x)
        x = self.chomp1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.ll_conv2(x)
        x = self.chomp2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        if self.skip_connection is not None:
            res = self.skip_connection(res)
        return x + res


class TemporalConvNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_channels, kernel_size=2, dropout=0.0):
        super(TemporalConvNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        layers = []
        self.num_levels = len(num_channels)

        for i in range(self.num_levels):
            dilation_size = 2 ** i
            in_channels = embedding_dim if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            setattr(
                self,
                "ll_temporal_block{}".format(i),
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout
                ),
            )

        self.fc = nn.Linear(num_channels[-1], vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)

        for i in range(self.num_levels):
            temporal_block = getattr(self, "ll_temporal_block{}".format(i))
            x = temporal_block(x)

        x = x[:, :, -1]
        x = self.fc(x)
        return x
