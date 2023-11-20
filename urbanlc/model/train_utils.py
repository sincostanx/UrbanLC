import random
import numpy as np
import copy
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

def set_seed(seed):
    # for some reasons, setting torch.backends.cudnn.deterministic = True will return an error
    # Can't pickle CudnnModule objects
    # ref: https://github.com/ray-project/ray/issues/8569
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    eval('setattr(torch.backends.cudnn, "benchmark", True)')
    eval('setattr(torch.backends.cudnn, "deterministic", True)')


def save_checkpoint(save_dir, name, model, optimizer, scheduler, epoch):
    temp_model = copy.deepcopy(model)
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"{name}.pt")
    state = {
        "model": temp_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
    }
    torch.save(state, filepath)
    return filepath


def load_checkpoint(
    checkpoint_path, model, optimizer=None, scheduler=None, device=torch.device("cuda")
):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_dict = checkpoint["model"]

    # check if the model_dict key match that of the model itself
    model.load_state_dict(model_dict)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    elapsed_epoch = checkpoint.get("epoch", 0)

    return model, optimizer, scheduler, elapsed_epoch


# modified from https://gitlab.com/winderl13/remote-sensing-landuse/-/blob/main/utils/data_preprocessing.py
# TODO: not sure if Pytorch natively support this op
def segment_satelite_image(
    img: torch.Tensor, sub_size: Optional[int] = 224, stride: Optional[int] = None
) -> Tuple[List[torch.Tensor], List[tuple]]:
    # format (N, C, W, H)
    assert isinstance(sub_size, int)
    if img.dim() == 3:
        if isinstance(img, torch.Tensor):
            img = torch.unsqueeze(img, dim=0)
        elif isinstance(img, np.ndarray):
            img = np.expand_dims(img, axis=0)
        else:
            raise TypeError("image must be Numpy array or Torch tensor")

    stride = sub_size if stride is None else stride
    data_out = []
    coordinate_out = []
    remain_w = img.shape[2] % stride
    remain_h = img.shape[3] % stride
    for i in range(0, img.shape[2] - sub_size, stride):
        for j in range(0, img.shape[3] - sub_size, stride):
            min_w, min_h = int(i), int(j)
            max_w, max_h = int(i + sub_size), int(j + sub_size)
            data_out.append(img[:, :, min_w:max_w, min_h:max_h])
            coordinate_out.append((min_w, max_w, min_h, max_h))

    if remain_h or remain_w:
        # remain_width
        i = img.shape[2] - sub_size
        for j in range(0, img.shape[3] - sub_size, stride):
            min_w, min_h = int(i), int(j)
            max_w, max_h = int(i + sub_size), int(j + sub_size)
            data_out.append(img[:, :, min_w:max_w, min_h:max_h])
            coordinate_out.append((min_w, max_w, min_h, max_h))

        # remain_height
        j = img.shape[3] - sub_size
        for i in range(0, img.shape[2] - sub_size, stride):
            min_w, min_h = int(i), int(j)
            max_w, max_h = int(i + sub_size), int(j + sub_size)
            data_out.append(img[:, :, min_w:max_w, min_h:max_h])
            coordinate_out.append((min_w, max_w, min_h, max_h))

        # the corner patch
        min_w, min_h = img.shape[2] - sub_size, img.shape[3] - sub_size
        max_w, max_h = img.shape[2], img.shape[3]
        data_out.append(img[:, :, min_w:max_w, min_h:max_h])
        coordinate_out.append((min_w, max_w, min_h, max_h))

    return data_out, coordinate_out


# naive implementation
# TODO: optimize this
def combine_prediction(
    preds: torch.Tensor,
    coordinates: List[Tuple[int]],
    original_size: Tuple[int],
    method: Optional[str] = "mean",
) -> torch.Tensor:
    # format (N, patch, 11, W, H)
    if method == "mean":
        # calculate mean probability of each pixel from patches and select the highest one
        # reduce memory consumption
        softmax = nn.Softmax(dim=0)
        combined_preds = []
        num_patches = preds.shape[0]
        for element in preds:
            output = torch.zeros(11, original_size[0], original_size[1])
            # actually the logic for calculating mean prob is wrong, but the order is still preserved anyway
            for _, (patch, bounds) in enumerate(zip(element, coordinates)):
                min_w, max_w, min_h, max_h = bounds
                output[:, min_w:max_w, min_h:max_h] = output[
                    :, min_w:max_w, min_h:max_h
                ] + (1.0 / num_patches) * (
                    softmax(patch) - output[:, min_w:max_w, min_h:max_h]
                )

            combined_preds.append(output)

        combined_preds = torch.stack(combined_preds, axis=0)
        return combined_preds
    else:
        raise NotImplementedError
