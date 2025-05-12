import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wz = nn.Linear(input_size, hidden_size)
        self.Uz = nn.Linear(hidden_size, hidden_size)
        self.Wr = nn.Linear(input_size, hidden_size)
        self.Ur = nn.Linear(hidden_size, hidden_size)
        self.Wh = nn.Linear(input_size, hidden_size)
        self.Uh = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h_prev):
        z_t = torch.sigmoid(self.Wz(x) + self.Uz(h_prev))
        r_t = torch.sigmoid(self.Wr(x) + self.Ur(h_prev))

        h_tilde = torch.tanh(self.Wh(x) + r_t * self.Uh(h_prev))  # 候选隐状态
        h_t = (1 - z_t) * h_prev + z_t * h_tilde  # 当前隐状态
        return h_t


class AttentionZeroedLayer(nn.Module):
    def __init__(self, threshold):
        super(AttentionZeroedLayer, self).__init__()
        self.threshold = threshold
        self.attention_weights = nn.Parameter(torch.ones(1))

    def forward(self, x):
        attention_scores = torch.sigmoid(self.attention_weights)
        zero_tensor = torch.zeros_like(x)
        x = torch.where(torch.abs(x) < self.threshold * attention_scores, zero_tensor, x)
        return x


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=0, dropout=0.0,
                 threshold=0.0):
        super(TemporalBlock, self).__init__()
        self.ll_conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                  dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.dropout1 = nn.Dropout(dropout)

        self.gru = CustomGRU(out_channels, out_channels)

        self.ll_conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                  dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.dropout2 = nn.Dropout(dropout)

        self.skip_connection = nn.Conv1d(in_channels, out_channels,
                                         kernel_size=1) if in_channels != out_channels else None

        # 添加 ZeroedLayer
        self.zeroed_layer = AttentionZeroedLayer(threshold)

    def forward(self, x):
        res = x
        x = self.ll_conv1(x)
        x = self.chomp1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = x.transpose(1,
                        2)  # (batch_size, out_channels, sequence_length) -> (batch_size, sequence_length, out_channels)

        # 初始化隐藏状态
        batch_size = x.size(0)
        h_prev = torch.zeros(batch_size, self.gru.hidden_size).to(x.device)  # 确保和输入在同一设备上

        # 逐步处理序列
        h_t = []
        for t in range(x.size(1)):
            h_prev = self.gru(x[:, t, :], h_prev)
            h_t.append(h_prev.unsqueeze(1))

        x = torch.cat(h_t, dim=1)
        x = x.transpose(1, 2)

        # 应用 ZeroedLayer
        x = self.zeroed_layer(x)

        x = self.ll_conv2(x)
        x = self.chomp2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        if self.skip_connection is not None:
            res = self.skip_connection(res)
        return x + res


class TemporalConvNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_channels, kernel_size=2, dropout=0.0, threshold=0.2):
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
                    dropout=dropout,
                    threshold=threshold
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
