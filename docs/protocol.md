# Fixed protocol

Historical freeze: `cora_official_last_20261029_163956`. Reference commit: `39a4089fe72ad9f055ed6fdb9746abdcfebc4d81`.

2708 nodes, 1433 features, 7 classes. Train 0–139 (20/class), validation 140–639, test 1708–2707; other 1068 nodes participate only through features/graph. Row-normalized features, binary undirected graph, one self-loop per node, symmetric degree normalization. Actual graph: 5278 undirected edges, 13264 normalized adjacency entries. Paper Table 1 reports 5429 edges; the difference has not been resolved edge by edge.

GCN/MLP: 1433→16→7, no bias, 23040 parameters, Xavier uniform, dropout 0.5. Adam lr=0.01, betas=(0.9,0.999), eps=1e-8, amsgrad=False, foreach=False, weight_decay=0. Training loss is mean CE on the 140 training labels + (5e-4/2) sum(W0²).

All seeds 0..9 were declared before formal evaluation. Maximum 200 epochs. Using zero-based epoch, stop if epoch>10 and current validation CE+L2 > mean(previous 10 validation objectives). Select `last.pt`; `best_val.pt` is diagnostic only. This follows the author's code, not a patience counter. Train/validation curves use post-update eval mode, dropout off. The test set does not select hyperparameters, seeds or checkpoints.

GCN mean=81.40%, sample std=0.64 pp; MLP mean=52.43%, sample std=10.89 pp (ddof=1). All 20 runs are retained, including MLP seed=0 test accuracy 21.6% after 22 epochs. Its validation CE fell while first-layer L2 rose, triggering the frozen rule. Paper fixed-split references are 81.5%/55.1% from 100 initializations; same-split std was not reported. Do not borrow the random-split standard error.
