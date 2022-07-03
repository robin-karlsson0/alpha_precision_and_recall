# Alpha Precision and Recall Metric
Function for computing the 'Alpha Precision and Recall' metric over two sets of features A and B.

The vanilla precision and recall (P&R) metric is sensitive to outliers that bloats the manifold estimation approximating distribution B. By removing a fraction $\alpha$ of features considered outliers, measured by smallest distance to a neighboring feature, a better estimation of the manifold and thus P&R values can be achieved.

Setting `alpha_interval=1` results in the vanilla P&R metric (default `alpha_interval=10`).

Ref: [Alaa A., van Breugel B., Saveliev E., van der Schaar M., How Faithful is your Synthetic Data? Sample-level Metrics for Evaluating and Auditing Generative Models, Arxiv, 2021](https://arxiv.org/abs/2102.08921)

## How to use

```
# Imports
from alpha_precision_recall import alpha_precision_recall

# 1. Compute set of features from a task encoder as a row-vector matrix
#    --> NOT included
feats_a = encoder(inputs_a)  # (N, D)
feats_b = encoder(inputs_b)  # (N, D)

2. Compute precision metric
results = alpha_precision_recall(feats_a, feats_b)

# results: List of tuples [(alpha, precision), ... ]
#
# Alpha | Precision
# -----------------
#   0.0 |      0.99 <-- Over all features
#   0.1 |      0.98 <-- Removed 10% of outlier features
#   0.2 |      0.97
#   ... |       ...

3. Compute recall metric
results = alpha_precision_recall(feats_b, feats_a)
```

