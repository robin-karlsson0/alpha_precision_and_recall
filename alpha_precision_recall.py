import numpy as np
import torch


def alpha_precision_recall(
        feats_a: torch.Tensor,
        feats_b: torch.Tensor,
        device: torch.device,
        max_sample_num: int = 10000,
        alpha_interval: int = 10,
        compute_mode: str = 'use_mm_for_euclid_dist_if_necessary') -> tuple:
    '''
    Function for computing the Alpha Precision and Recall metric over two sets
    of features representing distributions A and B.

    Removes outlier samples with most distant nearest neighbor samples.

    Computes recall when switching A and B positions.

    Ref: Alaa A, et al., How Faithful is your Synthetic Data? Sample-level
         Metrics for Evaluating and Auditing Generative Models, Arxiv, 2021
         https://arxiv.org/abs/2102.08921

    How to use:

        # 1. Compute set of features from a task encoder as a row-vector matrix
        feats_a = encoder(inputs_a)  # (N, D)
        feats_b = encoder(inputs_b)  # (N, D)

        # 2. Compute precision metric
        results = alpha_precision_recall(feats_a, feats_b)

        # 3. Compute recall metric
        results = alpha_precision_recall(feats_b, feats_a)

    Args:
        feats_a: Features from distribution A (N, D).
        feats_b: Features from comparison distribution B (M, D).
        device: Specify CPU or (some) GPU.
        max_sample_num: Maximum number of features for computing metric.
        alpha_interval: Number of alpha values to compute metric over.
            NOTE: Set to 1 to compute vanilla P&R metric over all samples.
        compute_mode: How to compute distances w. cdist() (approx. or exact).

    Returns:
        List of tuples [(alpha, precision), ... ]
    '''
    # Find number of features to use
    feats_a_num = feats_a.shape[0]
    feats_b_num = feats_b.shape[0]
    num_samples = min(feats_a_num, feats_b_num, max_sample_num)

    # Subsample features
    idxs = torch.randperm(feats_a_num)[:num_samples]
    feats_a = feats_a[idxs]
    idxs = torch.randperm(feats_b_num)[:num_samples]
    feats_b = feats_b[idxs]

    # Generate vector with nearest neighbor distance for allsamples
    # NOTE: Add diagonal values to remove distance to self
    dist_b2b = torch.cdist(feats_b, feats_b, compute_mode=compute_mode)
    dist_b2b += torch.eye(num_samples).to(device) * torch.max(dist_b2b)
    dist_b2nnb, _ = torch.min(dist_b2b, dim=1)  # (num_samples)

    # Compute P&R values over all alpha outlier ratios
    results = []  # Stores tuples (alpha, precision)
    for alpha in np.linspace(0, 1, alpha_interval, endpoint=False):

        # Remove fraction of outliers
        remove_num = int(alpha * num_samples)
        # Extract NOT removed samples by inversion
        num_keep = num_samples - remove_num
        _, keep_idxs = torch.topk(-dist_b2nnb, num_keep)
        # Sort indices to perserve order
        keep_idxs, _ = torch.sort(keep_idxs)
        feats_b_alpha = feats_b[keep_idxs]
        dist_b2nnb_alpha = dist_b2nnb[keep_idxs]

        # Randomly extract equal number of generated points
        idxs = torch.randperm(num_samples)[:num_keep]
        idxs, _ = torch.sort(idxs)
        feats_a_alpha = feats_a[idxs]

        # Compute 'a --> b' sample distances w. indexed by [a_idx, b_idx]
        dist_a2b_alpha = torch.cdist(feats_a_alpha, feats_b_alpha)

        # Condition satisfied if some dist(a --> b) < smallest dist(b --> b)
        # ==> [a_idx, b_idx] is negative value after subtracting KNN=1 distance
        #     for each sample 'b'
        dist_b2nnb_alpha = dist_b2nnb_alpha.unsqueeze(0).repeat(num_keep, 1)
        dist_diff = dist_a2b_alpha - dist_b2nnb_alpha

        # Compute ratio with 'a' samples satisfying condition
        dist_diff_min, _ = torch.min(dist_diff, dim=1)  # (num_keep)
        closeness_test = torch.zeros((num_keep))
        closeness_test[dist_diff_min < 0] = 1
        precision = torch.sum(closeness_test) / num_keep

        results.append((alpha, precision.item()))

    return results


if __name__ == "__main__":
    '''
    Example usage
    '''

    from torch.distributions.multivariate_normal import MultivariateNormal

    device = torch.device('cuda')
    feat_dim = 32
    feat_num = 1000

    print("< Same distribution >")

    distr_a = MultivariateNormal(torch.zeros(feat_dim), torch.eye(feat_dim))
    feats_a = distr_a.sample_n(feat_num)  # (N, D)
    distr_b = MultivariateNormal(torch.zeros(feat_dim), torch.eye(feat_dim))
    feats_b = distr_b.sample_n(feat_num)  # (N, D)

    feats_a = feats_a.to(device)
    feats_b = feats_b.to(device)

    results = alpha_precision_recall(feats_a, feats_b, device)

    alphas, precisions = list(zip(*results))
    print('Precision')
    for idx in range(len(alphas)):
        print(f'{alphas[idx]:.1f}, {precisions[idx]:.2f}')

    results = alpha_precision_recall(feats_b, feats_a, device)

    alphas, recalls = list(zip(*results))
    print('Recall')
    for idx in range(len(alphas)):
        print(f'{alphas[idx]:.1f}, {recalls[idx]:.2f}')

    print("\n< Distribution A is a subset of distribution B >")
    print('    ==> High precision (sample quality)')
    print('    ==> Low recall (sample diversity)')

    distr_a = MultivariateNormal(torch.zeros(feat_dim),
                                 0.5 * torch.eye(feat_dim))
    feats_a = distr_a.sample_n(feat_num)  # (N, D)
    distr_b = MultivariateNormal(torch.zeros(feat_dim), torch.eye(feat_dim))
    feats_b = distr_b.sample_n(feat_num)  # (N, D)

    feats_a = feats_a.to(device)
    feats_b = feats_b.to(device)

    results = alpha_precision_recall(feats_a, feats_b, device)

    alphas, precisions = list(zip(*results))
    print('Precision')
    for idx in range(len(alphas)):
        print(f'{alphas[idx]:.1f}, {precisions[idx]:.2f}')

    results = alpha_precision_recall(feats_b, feats_a, device)

    alphas, recalls = list(zip(*results))
    print('Recall')
    for idx in range(len(alphas)):
        print(f'{alphas[idx]:.1f}, {recalls[idx]:.2f}')
