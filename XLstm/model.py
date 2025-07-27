import torch
import torch.nn as nn
from xlstm.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig
from xlstm.blocks.mlstm.block import mLSTMBlockConfig
from xlstm.blocks.mlstm.layer import mLSTMLayerConfig

class xLSTMNet(nn.Module):
    def __init__(self, num_classes, num_embeddings, embedding_dim=64, hidden_size=128,
                 num_layers=1, dropout=0.1):
        super().__init__()

        self.num_embeddings = num_embeddings
        if self.num_embeddings > 0:
            self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        else:
            self.embedding = nn.Identity()

        # 构造 mLSTM block config，手动注入参数
        mlstm_layer_cfg = mLSTMLayerConfig(
            embedding_dim=embedding_dim,
            context_length=8,
            dropout=dropout,
            bias=True,
        )

        mlstm_block_cfg = mLSTMBlockConfig(mlstm=mlstm_layer_cfg)

        stack_config = xLSTMBlockStackConfig(
            mlstm_block=mlstm_block_cfg,
            slstm_block=None,
            num_blocks=num_layers,
            embedding_dim=embedding_dim,
            dropout=dropout,
            slstm_at=[],
            context_length=8
        )

        self.xlstm = xLSTMBlockStack(config=stack_config)
        self.classify = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        """
        :param x: LongTensor (batch_size, seq_len)
        :return: FloatTensor (batch_size, num_classes)
        """
        x = self.embedding(x)
        x = self.xlstm(x)
        x = x[:, -1, :]
        x = self.classify(x)
        return x
