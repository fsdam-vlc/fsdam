# metrics_llada.py
# Adopted metrics as in LLada code -https://github.com/yuchen2199/Explainable-Driver-Attention-Prediction/tree/main/utils/sal_metrics.py

import torch
import numpy as np
from audtorch.metrics.functional import pearsonr
from enum import Enum

EPS = 1e-7

def CC(pred: torch.Tensor, gt: torch.Tensor, eps: float = EPS):
    # pred, gt: [B,1,H,W]
    a = pearsonr(pred.flatten(), gt.flatten())
    a = torch.where(torch.isnan(a), torch.full_like(a, 0), a)
    a = a.mean()
    return float(a)

def KLDivergence(pred: torch.Tensor, gt: torch.Tensor, eps: float = EPS):
    # L1 normalize both, sum over all pixels
    P = pred
    P = P / (eps + torch.sum(P, dim=[1, 2, 3], keepdim=True))
    Q = gt
    Q = Q / (eps + torch.sum(Q, dim=[1, 2, 3], keepdim=True))
    R = Q * torch.log(eps + Q / (eps + P))
    kld = float(R.sum())
    return kld

def normalize_map(s_map: torch.Tensor):
    # per-image min-max to [0,1]
    batch_size = s_map.size(0)
    h = s_map.size(1)
    w = s_map.size(2)
    min_s_map = torch.min(s_map.view(batch_size, -1), 1)[0].view(batch_size, 1, 1).expand(batch_size, h, w)
    max_s_map = torch.max(s_map.view(batch_size, -1), 1)[0].view(batch_size, 1, 1).expand(batch_size, h, w)
    norm_s_map = (s_map - min_s_map) / (max_s_map - min_s_map * 1.0)
    return norm_s_map

def SIM(s_map: torch.Tensor, gt: torch.Tensor):
    # followed their normalize_map + L1 normalize + sum min
    s_map = s_map.squeeze(1)
    gt = gt.squeeze(1)
    batch_size = s_map.size(0)
    h = s_map.size(1)
    w = s_map.size(2)

    s_map_norm = normalize_map(s_map)
    gt_norm = normalize_map(gt)

    sum_s_map = torch.sum(s_map_norm.view(batch_size, -1), 1)
    expand_s_map = sum_s_map.view(batch_size, 1, 1).expand(batch_size, h, w)

    sum_gt = torch.sum(gt.view(batch_size, -1), 1)
    expand_gt = sum_gt.view(batch_size, 1, 1).expand(batch_size, h, w)

    s_map_norm = s_map_norm / (expand_s_map * 1.0)
    gt_norm = gt / (expand_gt * 1.0)

    s_map_norm = s_map_norm.view(batch_size, -1)
    gt_norm = gt_norm.view(batch_size, -1)
    return torch.sum(torch.min(s_map_norm, gt_norm), 1)  # per-sample vector

def discretize_gt(gt: np.ndarray, threshold: float = 0.7):
    gt = gt.astype(np.float32)
    epsilon = 1e-6
    binary_gt = np.where(gt >= threshold - epsilon, 1.0, 0.0)
    assert np.isin(binary_gt, [0, 1]).all(), "discretize error"
    return binary_gt

def AUC_J(s_map: torch.Tensor, gt: torch.Tensor):
    s_map = normalize_map(s_map.squeeze(1))
    s_map = s_map[0].cpu().detach().numpy()
    gt = normalize_map(gt.squeeze(1))
    gt = gt[0].cpu().detach().numpy()
    gt = discretize_gt(gt)

    thresholds = []
    for i in range(0, gt.shape[0]):
        for k in range(0, gt.shape[1]):
            if gt[i][k] > 0:
                thresholds.append(s_map[i][k])
    num_fixations = np.sum(gt)
    thresholds = sorted(set(thresholds))

    area = []
    area.append((0.0, 0.0))
    for thresh in thresholds:
        temp = np.zeros(s_map.shape)
        temp[s_map >= thresh] = 1.0
        assert np.max(gt) == 1.0
        assert np.max(s_map) == 1.0
        num_overlap = np.where(np.add(temp, gt) == 2)[0].shape[0]
        tp = num_overlap / (num_fixations * 1.0)
        fp = (np.sum(temp) - num_overlap) / ((np.shape(gt)[0] * np.shape(gt)[1]) - num_fixations)
        area.append((round(tp, 4), round(fp, 4)))
    area.append((1.0, 1.0))
    area.sort(key=lambda x: x[0])
    tp_list = [x[0] for x in area]
    fp_list = [x[1] for x in area]
    return np.trapz(np.array(tp_list), np.array(fp_list))

def AUC_B(s_map: torch.Tensor, gt: torch.Tensor, splits: int = 100, stepsize: float = 0.1):
    s_map = normalize_map(s_map.squeeze(1))
    s_map = s_map[0].cpu().detach().numpy()
    gt = normalize_map(gt.squeeze(1))
    gt = gt[0].cpu().detach().numpy()
    gt = discretize_gt(gt)
    num_fixations = np.sum(gt)

    num_pixels = s_map.shape[0] * s_map.shape[1]
    random_numbers = []
    for _ in range(0, splits):
        temp_list = []
        for _ in range(0, int(num_fixations)):
            temp_list.append(np.random.randint(num_pixels))
        random_numbers.append(temp_list)

    aucs = []
    for i in random_numbers:
        r_sal_map = []
        for k in i:
            r_sal_map.append(s_map[k % s_map.shape[0] - 1, k // s_map.shape[0]])
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        r_sal_map = np.array(r_sal_map)
        thresholds = sorted(set(thresholds))
        area = []
        area.append((0.0, 0.0))
        for thresh in thresholds:
            temp = np.zeros(s_map.shape)
            temp[s_map >= thresh] = 1.0
            num_overlap = np.where(np.add(temp, gt) == 2)[0].shape[0]
            tp = num_overlap / (num_fixations * 1.0)
            fp = len(np.where(r_sal_map > thresh)[0]) / (num_fixations * 1.0)
            area.append((round(tp, 4), round(fp, 4)))
        area.append((1.0, 1.0))
        area.sort(key=lambda x: x[0])
        tp_list = [x[0] for x in area]
        fp_list = [x[1] for x in area]
        aucs.append(np.trapz(np.array(tp_list), np.array(fp_list)))
    return np.mean(aucs)

def AUC_S(s_map: np.ndarray, gt: np.ndarray, other_map: np.ndarray, splits: int = 100, stepsize: float = 0.1):
    num_fixations = np.sum(gt)
    x, y = np.where(other_map == 1)
    other_map_fixs = []
    for j in zip(x, y):
        other_map_fixs.append(j[0] * other_map.shape[0] + j[1])
    ind = len(other_map_fixs)
    assert ind == np.sum(other_map), 'auc shuffle error'
    num_fixations_other = min(ind, num_fixations)

    num_pixels = s_map.shape[0] * s_map.shape[1]
    random_numbers = []
    for _ in range(0, splits):
        temp_list = []
        t1 = np.random.permutation(ind)
        for k in t1:
            temp_list.append(other_map_fixs[k])
        random_numbers.append(temp_list)

    aucs = []
    for i in random_numbers:
        r_sal_map = []
        for k in i:
            r_sal_map.append(s_map[k % s_map.shape[0] - 1, k / s_map.shape[0]])
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        r_sal_map = np.array(r_sal_map)
        thresholds = sorted(set(thresholds))
        area = []
        area.append((0.0, 0.0))
        for thresh in thresholds:
            temp = np.zeros(s_map.shape)
            temp[s_map >= thresh] = 1.0
            num_overlap = np.where(np.add(temp, gt) == 2)[0].shape[0]
            tp = num_overlap / (num_fixations * 1.0)
            fp = len(np.where(r_sal_map > thresh)[0]) / (num_fixations * 1.0)
            area.append((round(tp, 4), round(fp, 4)))
        area.append((1.0, 1.0))
        area.sort(key=lambda x: x[0])
        tp_list = [x[0] for x in area]
        fp_list = [x[1] for x in area]
        aucs.append(np.trapz(np.array(tp_list), np.array(fp_list)))
    return np.mean(aucs)

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
    def summary(self):
        if self.summary_type is Summary.NONE:
            return ""
        if self.summary_type is Summary.AVERAGE:
            return "{name} {avg:.3f}".format(**self.__dict__)
        if self.summary_type is Summary.SUM:
            return "{name} {sum:.3f}".format(**self.__dict__)
        if self.summary_type is Summary.COUNT:
            return "{name} {count:.3f}".format(**self.__dict__)
        raise ValueError("invalid summary type")
