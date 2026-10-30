"""Reuse the tensor conversion already checked in check_cora_tensors.py."""
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch


def load_cora_tensors(data_dir):
    """Load the verified processed files as CPU tensors."""
    data_dir = Path(data_dir)
    features_sp = sp.load_npz(data_dir / "features_norm.npz")
    adj_coo = sp.load_npz(data_dir / "adj_norm.npz").tocoo()

    x = torch.tensor(features_sp.toarray(), dtype=torch.float32)
    indices = torch.tensor(
        np.vstack((adj_coo.row, adj_coo.col)), dtype=torch.long
    )
    values = torch.tensor(adj_coo.data, dtype=torch.float32)
    adj_norm = torch.sparse_coo_tensor(
        indices, values, size=adj_coo.shape
    ).coalesce()

    with np.load(data_dir / "labels_and_masks.npz", allow_pickle=False) as saved:
        y = torch.tensor(saved["labels"], dtype=torch.long)
        train_mask = torch.tensor(saved["train_mask"], dtype=torch.bool)
        val_mask = torch.tensor(saved["val_mask"], dtype=torch.bool)
        test_mask = torch.tensor(saved["test_mask"], dtype=torch.bool)

    return x, adj_norm, y, train_mask, val_mask, test_mask
