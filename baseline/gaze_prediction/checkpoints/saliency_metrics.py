#saliency_metrics.py
import torch
import numpy as np

_EPS = 1e-6


def _tensor_to_np_safe(t: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy WITHOUT using tensor.numpy()."""
    return np.array(t.detach().cpu().tolist(), dtype=np.float32)



# ---------- CC ----------
def CC(pred: torch.Tensor, gt: torch.Tensor, eps: float = _EPS) -> float:
    """Pearson correlation (CC) between prediction and GT.

    Implemented in pure torch to avoid audtorch / numpy / numba issues.
    """
    x = pred.flatten().float()
    y = gt.flatten().float()

    # zero-mean
    x = x - x.mean()
    y = y - y.mean()

    # variances
    vx = (x * x).mean()
    vy = (y * y).mean()
    denom = (vx * vy).sqrt() + eps

    cc = (x * y).mean() / denom
    if torch.isnan(cc) or torch.isinf(cc):
        cc = torch.tensor(0.0, device=cc.device)
    return float(cc.item())


# ---------- KL divergence ----------
def KLDivergence(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-7) -> float:
    """
    KL(Q || P) between prediction and GT saliency maps.

    - Clamp to non-negative
    - L1-normalize per sample over [C,H,W]
    - Compute per-sample KL and then average
    - Replace NaN/inf by 0 so we never propagate NaNs
    """
    # flatten over spatial and channel
    P = pred.clone()
    Q = gt.clone()

    # clamp to non-negative
    P = torch.clamp(P, min=0.0)
    Q = torch.clamp(Q, min=0.0)

    # per-sample sums over all dims except batch
    dims = tuple(range(1, P.ndim))
    P_sum = P.sum(dim=dims, keepdim=True)
    Q_sum = Q.sum(dim=dims, keepdim=True)

    P = P / (P_sum + eps)
    Q = Q / (Q_sum + eps)

    # KL(Q||P) = sum Q * (log Q - log P)
    kl_map = Q * (torch.log(Q + eps) - torch.log(P + eps))
    kl_per_sample = kl_map.sum(dim=dims)

    # remove NaN/inf
    kl_per_sample = torch.nan_to_num(kl_per_sample, nan=0.0, posinf=0.0, neginf=0.0)

    return float(kl_per_sample.mean().item())



    
def _normalize_map_torch(s_map):
    # normalize the salience map (as done in MIT code)
    batch_size = s_map.size(0)
    h = s_map.size(1)
    w = s_map.size(2)

    min_s_map = torch.min(s_map.view(batch_size, -1), 1)[0].view(batch_size, 1, 1).expand(batch_size, h, w)
    max_s_map = torch.max(s_map.view(batch_size, -1), 1)[0].view(batch_size, 1, 1).expand(batch_size, h, w)

    denom = (max_s_map - min_s_map).clamp_min(1e-7)
    norm_s_map = (s_map - min_s_map) / denom
    return norm_s_map


def SIM(s_map, gt):
    '''MIT-standard Similarity metric (Similarity.m).
       s_map, gt: [B,1,H,W] or [B,H,W] tensors.
    '''
    s_map = s_map.squeeze(1)
    gt = gt.squeeze(1)
    batch_size = s_map.size(0)
    h = s_map.size(1)
    w = s_map.size(2)

    s_map_norm = _normalize_map_torch(s_map)
    gt_norm = _normalize_map_torch(gt)

    sum_s_map = torch.sum(s_map_norm.view(batch_size, -1), 1)
    expand_s_map = sum_s_map.view(batch_size, 1, 1).expand(batch_size, h, w)

    sum_gt = torch.sum(gt_norm.view(batch_size, -1), 1)
    expand_gt = sum_gt.view(batch_size, 1, 1).expand(batch_size, h, w)

    s_map_norm = s_map_norm / expand_s_map
    gt_norm = gt_norm / expand_gt

    s_map_norm = s_map_norm.view(batch_size, -1)
    gt_norm = gt_norm.view(batch_size, -1)
    return torch.sum(torch.min(s_map_norm, gt_norm), 1)
  




# ---------- helpers for AUC / NSS ----------
def _normalize_map_np(arr: np.ndarray) -> np.ndarray:
    """
    Minmax normalize a numpy array to [0,1]. If flat, returns zeros.
    """
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax - vmin < 1e-7:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - vmin) / (vmax - vmin)


def discretize_gt(
    gt: np.ndarray,
    threshold: float = 0.7,
    already_normalized: bool = False,
) -> np.ndarray:
    """
    Convert a continuous ground truth map to a binary fixation map.

    If already_normalized is False, we first minmax normalize gt to [0,1],
    then apply the threshold. If True, gt is assumed already in [0,1].
    """
    if not already_normalized:
        gt = _normalize_map_np(gt)

    gt = gt.astype(np.float32)
    epsilon = 1e-6
    binary_gt = np.where(gt >= threshold - epsilon, 1.0, 0.0)
    # no strict assertion on max; just ensure it is binary
    assert np.isin(binary_gt, [0, 1]).all(), "discretize_gt produced non-binary values"
    return binary_gt


# ---------- AUC-J ----------
def AUC_J(s_map: torch.Tensor, gt: torch.Tensor) -> float:
    """
    AUC-Judd. Classic implementation on top of continuous saliency map and
    binary fixation map derived from minmax-normalized GT.
    """
    # normalize saliency map to [0,1] per sample
    s_map_t = _normalize_map_torch(s_map.squeeze(1))  # [B,H,W]
    s_map_np = _tensor_to_np_safe(s_map_t[0])

    # normalize GT and binarize
    gt_t = _normalize_map_torch(gt.squeeze(1))
    gt_np = _tensor_to_np_safe(gt_t[0])
    gt_bin = discretize_gt(gt_np, threshold=0.7, already_normalized=True)

    num_fixations = np.sum(gt_bin)
    if num_fixations == 0:
        return 0.0

    thresholds = [s_map_np[i, k] for i in range(gt_bin.shape[0]) for k in range(gt_bin.shape[1]) if gt_bin[i, k] > 0]
    if len(thresholds) == 0:
        return 0.0

    thresholds = sorted(set(thresholds))

    area = [(0.0, 0.0)]
    for thresh in thresholds:
        temp = np.zeros_like(s_map_np, dtype=np.float32)
        temp[s_map_np >= thresh] = 1.0

        num_overlap = np.where(np.add(temp, gt_bin) == 2)[0].shape[0]
        tp = num_overlap / (num_fixations * 1.0)

        fp = (np.sum(temp) - num_overlap) / ((gt_bin.shape[0] * gt_bin.shape[1]) - num_fixations + 1e-6)
        area.append((round(tp, 4), round(fp, 4)))

    area.append((1.0, 1.0))
    area.sort(key=lambda x: x[0])
    tp_list = [x[0] for x in area]
    fp_list = [x[1] for x in area]
    return float(np.trapz(np.array(tp_list), np.array(fp_list)))


# ---------- AUC-B ----------
def AUC_B(
    s_map: torch.Tensor,
    gt: torch.Tensor,
    splits: int = 100,
) -> float:
    """
    AUC-Borji. Similar to MIT implementation, using random non-fixation locations.
    """
    s_map_t = _normalize_map_torch(s_map.squeeze(1))
    s_map_np = _tensor_to_np_safe(s_map_t[0])

    gt_t = _normalize_map_torch(gt.squeeze(1))
    gt_np = _tensor_to_np_safe(gt_t[0])
    gt_bin = discretize_gt(gt_np, threshold=0.7, already_normalized=True)

    num_fixations = np.sum(gt_bin)
    if num_fixations == 0:
        return 0.0

    num_pixels = s_map_np.shape[0] * s_map_np.shape[1]
    if num_pixels == 0:
        return 0.0

    # random indices for negative locations
    random_numbers = []
    for _ in range(splits):
        idxs = np.random.randint(num_pixels, size=int(num_fixations))
        random_numbers.append(idxs)

    aucs = []
    for idxs in random_numbers:
        r_sal_map = []
        for k in idxs:
            r_sal_map.append(s_map_np[k % s_map_np.shape[0] - 1, k // s_map_np.shape[0]])
        r_sal_map = np.array(r_sal_map)

        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        thresholds = sorted(set(thresholds))

        area = [(0.0, 0.0)]
        for thresh in thresholds:
            temp = np.zeros_like(s_map_np, dtype=np.float32)
            temp[s_map_np >= thresh] = 1.0

            num_overlap = np.where(np.add(temp, gt_bin) == 2)[0].shape[0]
            tp = num_overlap / (num_fixations * 1.0)

            fp = len(np.where(r_sal_map > thresh)[0]) / (num_fixations * 1.0 + 1e-6)
            area.append((round(tp, 4), round(fp, 4)))

        area.append((1.0, 1.0))
        area.sort(key=lambda x: x[0])
        tp_list = [x[0] for x in area]
        fp_list = [x[1] for x in area]

        aucs.append(np.trapz(np.array(tp_list), np.array(fp_list)))

    return float(np.mean(aucs)) if len(aucs) > 0 else 0.0


# ---------- NSS ----------
def NSS(s_map: torch.Tensor, gt: torch.Tensor) -> float:
    """
    Normalized Scanpath Saliency.

    1) Normalize prediction to zero mean and unit std.
    2) Binarize GT into fixations, using minmax-normalized GT with threshold 0.7.
    3) Average predicted saliency at fixation locations.

    This avoids the all-zero fixation mask problem and matches standard NSS logic.
    """
    s_map_np = _tensor_to_np_safe(s_map[0].squeeze(0))
    gt_np = _tensor_to_np_safe(gt[0].squeeze(0))

    # Normalize GT first, then threshold to create fixation mask
    gt_norm = _normalize_map_np(gt_np)
    gt_bin = discretize_gt(gt_norm, threshold=0.7, already_normalized=True)

    ys, xs = np.where(gt_bin == 1)
    if len(xs) == 0:
        return 0.0

    s_map_norm = (s_map_np - np.mean(s_map_np)) / (np.std(s_map_np) + 1e-7)
    values = s_map_norm[ys, xs]
    if values.size == 0:
        return 0.0
    return float(np.mean(values))
