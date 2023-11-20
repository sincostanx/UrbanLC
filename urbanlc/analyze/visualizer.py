import os
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import torch
from .constant import LANDSAT_RGB, ESA2021_LABEL
import ffmpeg

sns.set_theme()

# http://hydro.iis.u-tokyo.ac.jp/~akira/page/Python/contents/plot/color/colormap.html
ESA_color = ListedColormap([v[0] for k, v in ESA2021_LABEL.items()])
ESA_color.set_under("black")
ESA_color.set_over("black")
bounds = list(ESA2021_LABEL.keys()) + [101]
norm = BoundaryNorm(bounds, ESA_color.N)


def plot_class_distribution(
    label: np.ndarray,
    normalize: Optional[bool] = True,
    outfile: Optional[str] = None,
    figsize: Optional[Tuple[int]] = (6, 3),
) -> None:
    label = label.flatten()
    fig, ax = plt.subplots(figsize=figsize)

    count = Counter(label)
    df = pd.DataFrame.from_dict(count, orient="index").sort_index()
    if normalize:
        df[0] = df[0] / df[0].sum()

    df.plot(kind="bar", ax=ax)
    ax.get_legend().remove()
    ax.set_ylim([0, 1])
    ax.set_title("Class distribution")
    fig.autofmt_xdate(rotation=0, ha="center")

    for p in ax.patches:
        ax.annotate(
            str(round(p.get_height(), 2)), (p.get_x() * 1.005, p.get_height() * 1.05)
        )

    if outfile is not None:
        os.makedirs(Path(outfile).parent, exist_ok=True)
        fig.savefig(outfile, dpi=300)


def show_esa_label() -> None:
    labels = np.array([list(ESA2021_LABEL.keys())])
    descriptions = [v[1] for v in ESA2021_LABEL.values()]

    fig, ax = plt.subplots(figsize=(300, 1))
    ax.imshow(labels, cmap=ESA_color, norm=norm)

    ax.set_xticks(np.arange(len(descriptions)))
    ax.set_xticklabels(descriptions, fontsize=10)
    ax.grid(False)
    ax.get_yaxis().set_visible(False)

    ax2 = ax.secondary_xaxis("top")
    ax2.set_xticks(np.arange(len(descriptions)))
    ax2.set_xticklabels(labels[0], fontsize=10)
    ax2.tick_params(top=False)


def plot_land_cover(
    img: np.ndarray,
    ax: plt.Axes,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    if img.shape[0] == 1:
        img = img.transpose(1, 2, 0)

    ax.imshow(img, cmap=ESA_color, norm=norm)
    ax.grid(False)
    ax.axis("off")
    if title is not None:
        assert isinstance(title, str)
        ax.set_title(title, fontsize=14, fontweight="bold")

    if save_path is not None:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()


def plot_landsat(
    img,
    dataset: str,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    ax: plt.Axes = None,
) -> None:
    assert dataset in LANDSAT_RGB
    assert ax is not None
    if np.argmin(img.shape) == 0:
        img = img.transpose(1, 2, 0)

    ax.imshow(img[:, :, LANDSAT_RGB[dataset]])
    ax.grid(False)
    ax.axis("off")
    if title is not None:
        assert isinstance(title, str)
        ax.set_title(title, fontsize=14, fontweight="bold")

    if save_path is not None:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()


def plot_change(
    img_paths: Optional[List[str]] = None,
    root: Optional[str] = None,
    framerate: Optional[float] = 1,
    save_path: Optional[str] = "output.mp4",
) -> None:
    assert not ((img_paths is None) and (root is None))
    if os.path.exists(save_path):
        os.remove(save_path)

    if root is not None:
        input = os.path.join(root, "*.png")
        (
            ffmpeg.input(input, pattern_type="glob", framerate=framerate)
            .output(save_path)
            .run()
        )
    else:
        content = [f"file {img_path}'\n" for img_path in img_paths]
        with open("temp.txt", "w") as f:
            f.writelines(content)

        # https://stackoverflow.com/questions/67151665/can-i-pass-a-list-of-image-into-the-input-method-of-ffmpeg-python
        (
            ffmpeg.input("temp.txt", r=str(framerate), f="concat", safe="0")
            .output(save_path)
            .run()
        )

        os.remove("temp.txt")


def visualize_data_batch(
    images: torch.Tensor, gts: torch.Tensor, dataset: str, ax: plt.Axes
) -> None:
    for i, (image, gt) in enumerate(zip(images, gts)):
        plot_landsat(image.numpy(), dataset=dataset, ax=ax[2 * i])
        plot_land_cover(gt.numpy(), ax=ax[2 * i + 1])
