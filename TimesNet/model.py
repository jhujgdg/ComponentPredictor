from TimesNet import Model
import torch
import torch.nn as nn
import torch.nn.functional as F

class TimesNetClassifierWrapper(nn.Module):
    def __init__(self, base_model: Model, num_embeddings: int):
        super().__init__()
        self.base_model = base_model
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, base_model.configs.enc_in)
        self.register_buffer("mask_all_ones", torch.ones(1))

    def forward(self, x):
        x_embed = self.embedding(x)  # -> (B, T, embedding_dim)
        x_mark_enc = torch.ones_like(x, dtype=torch.float32)
        return self.base_model.classification(x_embed, x_mark_enc)
