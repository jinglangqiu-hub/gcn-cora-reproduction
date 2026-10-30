"""Two-layer node-wise MLP, matched to the GCN except for graph propagation."""
import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.5):
        super().__init__()
        # Keep the same (input, output) weight layout and Xavier draw order as GCN.
        self.w0 = nn.Parameter(torch.empty(in_features, hidden_features))
        self.w1 = nn.Parameter(torch.empty(hidden_features, out_features))
        nn.init.xavier_uniform_(self.w0)
        nn.init.xavier_uniform_(self.w1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = F.relu(x @ self.w0)
        x = self.dropout(x)
        return x @ self.w1
