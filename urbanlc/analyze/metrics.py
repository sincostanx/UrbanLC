# ref: http://gsp.humboldt.edu/olm/Courses/GSP_216/lessons/accuracy/metrics.html
import numpy as np
import os
import rasterio
import torch
from torchmetrics import ConfusionMatrix

from ..utils import open_at_size, open_at_scale
from .constant import ESA1992_map, ESA2021_map, ESA2021_CLASSES

def confusion_matrix(
    pred_path,
    gt_path,
    mapper_gt = ESA1992_map,
    mapper_pred = ESA2021_map,
    gt_downscale_factor = None,
    use_pred_as_ref=False,
):
    assert os.path.exists(pred_path)
    if not os.path.exists(gt_path):
        return None
    else:
        if not use_pred_as_ref:
            if gt_downscale_factor is not None:
                gt = open_at_scale(gt_path, gt_downscale_factor)
            else:
                gt = rasterio.open(gt_path).read()
            pred = open_at_size(pred_path, gt)
        else:
            print("Here")
            if gt_downscale_factor is not None:
                pred = open_at_scale(pred_path, gt_downscale_factor)
            else:
                pred = rasterio.open(pred_path).read()
            gt = open_at_size(gt_path, pred)

        assert gt.shape == pred.shape

        gt = np.vectorize(lambda x: mapper_gt[x])(gt)
        pred = np.vectorize(lambda x: mapper_pred[x])(pred)

        gt = torch.from_numpy(gt)
        pred = torch.from_numpy(pred)
        assert gt.shape == pred.shape

        CONFUSION_MATRIX = ConfusionMatrix(
            task="multiclass",
            num_classes=len(set(list(mapper_pred.values()))),
            ignore_index=-1,
        )
        return CONFUSION_MATRIX(pred, gt).numpy().transpose()
    
def accuracy(m):
    return m.diagonal().sum() / m.sum()

def producer_accuracy(m):
    return m.diagonal() / m.sum(axis=0)

def user_accuracy(m):
    return m.diagonal() / m.sum(axis=1)

def cohen_kappa(m):
    n = m.sum()

    row_totals = m.sum(axis=1)
    col_totals = m.sum(axis=0)

    p0 = np.trace(m) / n
    pe = row_totals * col_totals / (n ** 2)
    kappa = (p0 - pe.sum()) / (1 - pe.sum())
    return kappa


def get_class_distribution(
    path,
    downsample_scale,
    indices=ESA2021_CLASSES,
):
    data = open_at_scale(path, downsample_scale=downsample_scale).flatten()
    dist = [
        len(data[data == index]) / len(data) for index in indices
    ]
    return dist

# m = [[21, 6, 0], [5, 31, 1], [7, 2, 22]]
# m = np.array(m)
# print(accuracy(m))
# print(producer_accuracy(m))
# print(user_accuracy(m))