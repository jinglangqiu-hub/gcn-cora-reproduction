from torch import nn
from torch.nn import functional as F

from .gcn_layer import GraphConvolution


class GCN(nn.Module):
    def __init__(
        self, in_features, hidden_features, out_features, dropout=0.5
    ):
        super().__init__()

        self.gc1 = GraphConvolution(in_features, hidden_features)
        self.gc2 = GraphConvolution(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_norm):
        x = self.dropout(x)
        x = F.relu(self.gc1(x, adj_norm))
        x = self.dropout(x)
        return self.gc2(x, adj_norm)