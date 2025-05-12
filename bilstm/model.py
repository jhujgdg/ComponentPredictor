import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        assert (
                self.head_dim * num_heads == embed_dim
        ), "Embedding dimension must be divisible by number of heads"

        self.values = nn.Linear(embed_dim, embed_dim, bias=False)
        self.keys = nn.Linear(embed_dim, embed_dim, bias=False)
        self.queries = nn.Linear(embed_dim, embed_dim, bias=False)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        N, seq_length, embed_dim = x.shape
        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)

        # Split into multiple heads
        values = values.view(N, seq_length, self.num_heads, self.head_dim)
        keys = keys.view(N, seq_length, self.num_heads, self.head_dim)
        queries = queries.view(N, seq_length, self.num_heads, self.head_dim)

        values = values.permute(0, 2, 1, 3)  # (N, num_heads, seq_length, head_dim)
        keys = keys.permute(0, 2, 1, 3)  # (N, num_heads, seq_length, head_dim)
        queries = queries.permute(0, 2, 1, 3)  # (N, num_heads, seq_length, head_dim)

        energy = torch.einsum("nqhd,nkhd->nqhk", [queries, keys])  # (N, num_heads, seq_length, seq_length)
        attention = F.softmax(energy / (self.embed_dim ** (1 / 2)), dim=3)  # scaled dot-product attention

        out = torch.einsum("nqhk,nvhd->nqhd", [attention, values]).reshape(N, seq_length,
                                                                           self.embed_dim)  # (N, seq_length, embed_dim)
        return self.fc_out(out)


class EnhancedBiLSTM(nn.Module):
    def __init__(self, args):
        super(EnhancedBiLSTM, self).__init__()
        self.args = args
        self.hidden_dim = args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num

        self.embed = nn.Embedding(V, D, padding_idx=args.paddingId)
        if args.word_Embedding:
            self.embed.weight.data.copy_(args.pretrained_weight)

        self.bilstm = nn.LSTM(D, self.hidden_dim // 2, num_layers=self.num_layers,
                              dropout=0.5 if self.num_layers > 1 else 0,
                              bidirectional=True, bias=False)

        self.attention = MultiHeadAttention(embed_dim=self.hidden_dim, num_heads=8)
        self.dropout = nn.Dropout(0.5)
        self.hidden2label1 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.hidden2label2 = nn.Linear(self.hidden_dim // 2, C)

    def forward(self, x):
        embed = self.embed(x)
        x = embed.view(len(x), embed.size(1), -1)
        bilstm_out, _ = self.bilstm(x)  # (seq_length, batch_size, hidden_dim * 2)

        attn_out = self.attention(bilstm_out)  # (seq_length, batch_size, hidden_dim)

        attn_out = attn_out[:, -1, :]
        attn_out = self.dropout(attn_out)
        y = self.hidden2label1(attn_out)
        y = self.hidden2label2(y)
        logit = y
        return logit
