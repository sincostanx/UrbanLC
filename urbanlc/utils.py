import rasterio
import os
from pathlib import Path
import numpy as np
from typing import Optional, Dict, Any
from rasterio.enums import Resampling

def open_at_size(
    path: str,
    ref: np.ndarray,
):
    """
    Open .tif file and downsample it to match the size of another tif file
    """
    with rasterio.open(path) as dataset:
        data = dataset.read(
            out_shape = ref.shape,
            resampling=Resampling.mode
        )
    
    return data

def open_at_scale(
    path: str,
    downsample_scale: float,
):
    """
    Open .tif file and downsample it by a constant factor
    """
    with rasterio.open(path) as dataset:
        data = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height / downsample_scale),
                int(dataset.width / downsample_scale)
            ),
            resampling=Resampling.mode
        )
    
    return data

def export_geotiff(
    img: np.ndarray,
    save_path: str,
    output_meta: Dict[str, Any],
    compress: Optional[str] = None,
    tiled: Optional[bool] = True,
    blockxsize: Optional[int] = 256,
    blockysize: Optional[int] = 256,
    interleave: Optional[str] = "band",
) -> None:
    output_meta.update({
        "driver": "GTiff",
        "count": img.shape[0],
        "height": img.shape[1],
        "width": img.shape[2],
    })
    try:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        with rasterio.open(
            save_path,
            "w",
            compress=compress,
            tiled=tiled,
            blockxsize=blockxsize,
            blockysize=blockysize,
            interleave=interleave,
            **output_meta
        ) as f:
            f.write(img)
    except Exception as e:
        raise (e)
