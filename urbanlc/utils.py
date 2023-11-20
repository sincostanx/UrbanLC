import os
from tqdm.auto import tqdm

import rioxarray as rio
import xarray as xr

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
# from utils import export_geotiff
from typing import List, Optional, Tuple

"""
CONFUSION

ESA 2019

[10] Cropland, rainfed
	[11] Herbaceous cover
	[12] Tree or shrub cover
[20] Cropland, irrigated or post-flooding

[30] Mosaic cropland (>50%) / natural vegetation (tree, shrub, herbaceous cover) (<50%)
[40] Mosaic natural vegetation (tree, shrub, herbaceous cover) (>50%) / cropland (<50%)
[100] Mosaic tree and shrub (>50%) / herbaceous cover (<50%)
[110] Mosaic herbaceous cover (>50%) / tree and shrub (<50%)
[180] Shrub or herbaceous cover, flooded, fresh/saline/brakish water

ESA 2021

[20] Shrubland
[40] Cropland
[90] Herbaceous wetland
[95] Mangroves
"""

# https://maps.elie.ucl.ac.be/CCI/viewer/download/CCI-LC_Maps_Legend.pdf?fbclid=IwAR0R40UipA-dLBCiAQf-TxjYDA8-f35wlafkQ5XkLn4nVGDpuNIxnKli9F8
# https://climate.esa.int/media/documents/ESACCI-LC-Ph2-PUGv3_1.1.pdf
# https://worldcover2021.esa.int/data/docs/WorldCover_PUM_V2.0.pdf
ESA1992_ESA2021_map = {
    0: -1,                    # No Data
    10: 40,                   # Cropland, rainfed
        11: 40,                   # Herbaceous cover
        12: 40,                   # Tree or shrub cover
    20: 40,                   # Cropland, irrigated or post‐flooding
    30: 40,                   # Mosaic cropland (>50%) / natural vegetation (tree, shrub, herbaceous cover) (<50%)
    40: 20,                   # Mosaic  natural vegetation (tree, shrub, herbaceous cover) (>50%) / cropland (<50%)
    50: 10,                   # Tree cover, broadleaved, evergreen, closed to open (>15%)
    60: 10,                   # Tree cover, broadleaved, deciduous, closed to open (>15%)
        61: 10,                   # Tree cover, broadleaved, deciduous, closed (>40%)
        62: 10,                   # Tree cover, broadleaved, deciduous, open (15‐40%)
    70: 10,                   # Tree cover, needleleaved, evergreen, closed to open (>15%)
        71: 10,                   # Tree cover, needleleaved, evergreen, closed (>40%)
        72: 10,                   # Tree cover, needleleaved, evergreen, open (15‐40%)
    80: 10,                   # Tree cover, needleleaved, deciduous, closed to open (>15%)
        81: 10,                   # Tree cover, needleleaved, deciduous, closed (>40%)
        82: 10,                   # Tree cover, needleleaved, deciduous, open (15‐40%)
    90: 10,                   # Tree cover, mixed leaf type (broadleaved and needleleaved)
    100: 20,                  # Mosaic tree and shrub (>50%) / herbaceous cover (<50%)
    110: 20,                  # Mosaic herbaceous cover (>50%) / tree and shrub (<50%)
    120: 20,                  # Shrubland
        121: 20,                  # Evergreen shrubland
        122: 20,                  # Deciduous shrubland
    130: 30,                  # Grassland
    140: 100,                  # Lichens and mosses
    150: 60,                  # Sparse vegetation (tree, shrub, herbaceous cover) (<15%)
        151: 60,                  # Sparse tree (<15%)
        152: 60,                  # Sparse shrub (<15%)
        153: 60,                  # Sparse herbaceous cover (<15%)
    160: 10,                  # Tree cover, flooded, fresh or brakish water
    170: 10,                  # Tree cover, flooded, saline water
    180: 90,                  # Shrub or herbaceous cover, flooded, fresh/saline/brakish water
    190: 50,                  # Urban areas
    200: 60,                  # Bare areas
        201: 60,                  # Consolidated bare areas
        202: 60,                  # Unconsolidated bare areas
    210: 80,                  # Water bodies
    220: 70,                  # Permanent snow and ice
}


# modified from https://gitlab.com/winderl13/remote-sensing-landuse/-/blob/Landcover/NctoTiff.py
def export_nc2tif(filename: str, save_path: str):
    try:
        nc_file = xr.open_dataset(filename)

        LC = nc_file["lccs_class"]
        LC = LC.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
        LC.rio.write_crs("epsg:4326", inplace=True)
        LC.rio.to_raster(save_path)
    except FileNotFoundError:
        print("error")


def convert_nc2tif(root: str, start: int, end: int, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)

    valid_range = range(max(1992, start), min(2016, end + 1))
    for year in tqdm(valid_range):
        filename = os.path.join(
            root, f"ESACCI-LC-L4-LCCS-Map-300m-P1Y-{year}-v2.0.7cds.nc"
        )
        save_path = os.path.join(outdir, f"{year}.tif")
        export_nc2tif(filename, save_path)

    valid_range = range(max(2016, start), min(2021, end + 1))
    for year in tqdm(valid_range):
        filename = os.path.join(root, f"C3S-LC-L4-LCCS-Map-300m-P1Y-{year}-v2.1.1.nc")
        save_path = os.path.join(outdir, f"{year}.tif")
        export_nc2tif(filename, save_path)


def extract_patch(
    filename: str,
    bounds: List[float],
    original_size: Tuple[int],
    save_path: Optional[str] = None,
) -> np.ndarray:
    # bounds = [W, S, E, N] = [left, bottom, right, top]
    with rasterio.open(filename) as src:
        gt = src.read(
            1,
            window=from_bounds(*bounds, src.transform),
            out_shape=(src.count, original_size[0], original_size[1]),
            resampling=Resampling.nearest,
        )

    gt = np.expand_dims(gt, axis=0)
    if save_path is not None:
        output_meta = rasterio.open(filename).meta.copy()
        export_geotiff(gt, save_path, output_meta)

    return gt

import rasterio
import os
from pathlib import Path
import numpy as np
from typing import Optional, Dict, Any
from rasterio.enums import Resampling
import numbers

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
    Open a TIFF (.tif) file and downsample it by a constant factor.

    Parameters
    ----------
    path : str
        The path to the TIFF file to be opened.

    downsample_scale : float
        The constant factor by which the TIFF file will be downsampled.
        Should be a positive floating-point number greater than 0.

    Returns
    -------
    numpy.ndarray
        The downsampled data from the TIFF file.

    Raises
    ------
    rasterio.errors.RasterioIOError
        If there is an issue opening the TIFF file.

    Notes
    -----
    This function uses the rasterio library to open TIFF files and performs
    downsampling using mode.

    Examples
    --------
    >>> data = open_at_scale('/path/to/example.tif', 0.5)
    >>> print(data.shape)
    (bands, new_height, new_width)

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


if __name__ == "__main__":
    convert_nc2tif("./", 1992, 1992, "./")
