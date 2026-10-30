import math

import torch
from torch import nn


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(in_features, out_features)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj_norm):
        support = x @ self.weight
        output = torch.sparse.mm(adj_norm, support)
        return output
if __name__ == "__main__":
    X = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])

    # 已添加自环的邻接矩阵
    A_tilde = torch.tensor([
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ])

    degree = A_tilde.sum(dim=1)
    inverse_sqrt = degree.rsqrt()

    A_dense = (
        inverse_sqrt[:, None]
        * A_tilde
        * inverse_sqrt[None, :]
    )
    A_sparse = A_dense.to_sparse_coo().coalesce()

    layer = GraphConvolution(2, 1)

    # 验算时固定权重，方便与手算比较
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0], [2.0]]))

    output = layer(X, A_sparse)

    c = 1 / math.sqrt(6)
    expected = torch.tensor([
        [0.5 + 2 * c],
        [2 / 3 + 4 * c],
        [1.5 + 2 * c],
    ])

    torch.testing.assert_close(
        output, expected, rtol=1e-5, atol=1e-6
    )

    print("Output:", output.detach().squeeze())
    print("Expected:", expected.squeeze())
        # 教学用标量，只用于检查梯度
    output.sum().backward()

    # 用稠密矩阵运算作为梯度对照
    weight_dense = (
        layer.weight.detach().clone().requires_grad_(True)
    )
    output_dense = A_dense @ (X @ weight_dense)
    output_dense.sum().backward()

    torch.testing.assert_close(output, output_dense)
    torch.testing.assert_close(
        layer.weight.grad, weight_dense.grad
    )
    assert torch.isfinite(layer.weight.grad).all()

    print("Weight gradient:", layer.weight.grad)
    print("Layer forward and gradient checks passed.")