"""
modified from 
https://github.com/microsoft/torchgeo/blob/main/torchgeo/datasets/landsat.py
https://github.com/microsoft/torchgeo/blob/main/torchgeo/datasets/geo.py
"""
import glob
import os
import sys
from typing import Dict, List, Tuple, cast, Optional
from natsort import natsorted

import rasterio
from rasterio.crs import CRS
from rasterio.vrt import WarpedVRT

from torch.utils.data import DataLoader
from torchgeo.datasets import GeoDataset, RasterDataset, stack_samples
from torchgeo.samplers import RandomGeoSampler

from train_utils import set_seed
import numpy as np
import random

set_seed(0)

def parse_paths(root: str, filename_glob: str, exclude: Optional[List[str]] = None):
    pathname = os.path.join(root, filename_glob)
    all_paths = list(glob.glob(pathname, recursive=True))

    # naive implementation
    if exclude is not None:
        if isinstance(exclude, str):
            exclude = [exclude]
        for keyword in exclude:
            all_paths = [val for val in all_paths if keyword not in val]

    return natsorted(all_paths)

# https://stackoverflow.com/questions/62744176/how-to-overwrite-parent-class-init-method-and-use-super-to-call-grandparent-ini
class CustomRasterDataset(RasterDataset):
    def __init__(self, root=None, crs=None, res=None, bands=None, transforms=None, cache=False, exclude=None, all_paths=None):
        GeoDataset.__init__(self, transforms)

        self.root = root
        self.cache = cache

        if all_paths is None:
            all_paths = parse_paths(root, self.filename_glob, exclude)
        else:
            all_paths = natsorted(all_paths)
        print(len(all_paths))

        # Populate the dataset index
        i = 0
        for filepath in all_paths:
            try:
                with rasterio.open(filepath) as src:
                    # See if file has a color map
                    if len(self.cmap) == 0:
                        try:
                            self.cmap = src.colormap(1)
                        except ValueError:
                            pass

                    if crs is None:
                        crs = src.crs
                        self.transform = src.transform
                    if res is None:
                        res = src.res[0]

                    with WarpedVRT(src, crs=crs) as vrt:
                        minx, miny, maxx, maxy = vrt.bounds
            except rasterio.errors.RasterioIOError:
                continue
            else:
                mint: float = 0
                maxt: float = sys.maxsize

                coords = (minx, maxx, miny, maxy, mint, maxt)
                self.index.insert(i, coords, filepath)
                i += 1

        if i == 0:
            raise FileNotFoundError(
                f"No {self.__class__.__name__} data was found in '{root}'"
            )

        if bands and self.all_bands:
            band_indexes = [self.all_bands.index(i) + 1 for i in bands]
            self.bands = bands
            assert len(band_indexes) == len(self.bands)
        elif bands:
            msg = (
                f"{self.__class__.__name__} is missing an `all_bands` attribute,"
                " so `bands` cannot be specified."
            )
            raise AssertionError(msg)
        else:
            band_indexes = None
            self.bands = self.all_bands

        self.band_indexes = band_indexes
        self._crs = cast(CRS, crs)
        self.res = cast(float, res)

class Landsat(CustomRasterDataset):
    separate_files = False
    def __init__(self, root, filename_glob, all_bands, *args, **kwargs):
        self.all_bands = all_bands
        self.filename_glob = filename_glob
        self.is_image = True

        super().__init__(root, *args, **kwargs)

    @classmethod
    def MSS(cls, root=None, filename_glob=None, *args, **kwargs):
        all_bands = ["B4", "B5", "B6", "B7"]
        return Landsat(root, filename_glob, all_bands, *args, **kwargs)

    @classmethod
    def TM(cls, root=None, filename_glob=None, *args, **kwargs):
        all_bands = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7", "ST_B6"]
        return Landsat(root, filename_glob, all_bands, *args, **kwargs)

    @classmethod
    def OLITIRS(cls, root=None, filename_glob=None, *args, **kwargs):
        all_bands = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "ST_B10"]
        return Landsat(root, filename_glob, all_bands, *args, **kwargs)

class ESA2021(CustomRasterDataset):
    separate_files = False
    def __init__(self, root=None, filename_glob=None, crs=None, res=None, bands=None, transforms=None, cache=True, exclude=None, all_paths=None):
        self.all_bands = ["Map"]
        self.filename_glob = filename_glob
        self.is_image = False

        super().__init__(root, crs, res, bands, transforms, cache, exclude, all_paths)

# root, filename_glob, exclude, all_paths
def get_dataloader(landsat, esa, sensor_type, tile_per_loader=24, input_size=(224, 224), length=25000, batch_size=64, num_workers=8):
    assert sensor_type in ["MSS", "TM", "OLITIRS"], "valid sensor_type are ['MSS', 'TM', 'OLITIRS']"
    
    all_img_paths = natsorted(parse_paths(**landsat))
    all_label_paths = natsorted(parse_paths(**esa))
    assert len(all_img_paths) == len(all_label_paths)

    idxs = list(range(len(all_img_paths)))
    random.shuffle(idxs)

    loaders = []
    TILE_PER_LOADER = tile_per_loader
    num_loaders = int(np.ceil(float(len(idxs)) / TILE_PER_LOADER))
    for i in range(num_loaders):
        selected_idx = idxs[i*TILE_PER_LOADER : min(len(idxs), (i+1)*TILE_PER_LOADER)]
        print(selected_idx)
        selected_img = np.array(all_img_paths)[selected_idx]
        selected_gt = np.array(all_label_paths)[selected_idx]

        img_dataset = getattr(Landsat, sensor_type)(all_paths=selected_img, cache=False)
        label_dataset = ESA2021(all_paths=selected_gt, cache=True)
        dataset = img_dataset & label_dataset

        sampler = RandomGeoSampler(dataset, size=input_size, length=length)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=stack_samples, \
            shuffle=False, num_workers=num_workers, pin_memory=False)
        loaders.append(loader)

    return loaders