from typing import List, Optional, Type, Union, Tuple
import torch 
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import os
import tifffile
import csv
from src.utils import utils as Util


def _is_visible_tif(path: Path) -> bool:
    return path.suffix.lower() in {'.tif', '.tiff'} and not path.name.startswith('.')


class GeoDatasetSuperRes(torch.utils.data.Dataset):
    """Dataset for Super-Resolution with ESM simulation and ERA5 reanalysis"""
    
    def __init__(self, stage: str, config, epsilon: float = 0.0001):
        self.stage = stage
        self.config = config
        self.epsilon = epsilon
        raw_index_csv = getattr(config, "index_csv", None)
        self.index_csv = raw_index_csv.strip() if isinstance(raw_index_csv, str) else raw_index_csv

        self.hr_path = config.data_path  # High resolution path
        self.lr_path = config.data_path_lowres

        assert stage in ['train', 'valid', 'test'], "stage needs to be train, valid or test"
        
        if self.index_csv:
            print(f"Using index CSV for data selection: {self.index_csv}")
            
            index_csv_path = self.index_csv
            # Assuming same filenames for both HR and LR versions

            self.filenames = self._read_stems_from_csv(index_csv_path)
            
            # self.filenames = self._build_paths_from_stems(config.data_path_lowres, stems)
            # self.hr_path = self._build_paths_from_stems(config.data_path, stems)

        else:
        
            # Assuming same filenames for both HR and LR versions
            self.filenames = sorted(
                filename for filename in os.listdir(self.hr_path)
                if not filename.startswith('.')
            )
        
        self.filenames_test = sorted(
            filename for filename in os.listdir(config.data_path_test)
            if not filename.startswith('.')
        )
        
        self.train_split = 0.05
        self.splits = {
            "train": self.filenames[0:int(len(self.filenames)*(1-self.train_split))],
            "valid": self.filenames[int(len(self.filenames)*(1-self.train_split)):],
            "test": self.filenames_test
        }
        
        self.pad = torch.nn.ZeroPad2d(config.pad_input)
        
        # Super-resolution factor
        # self.scale_factor = getattr(config, 'scale_factor', 4)  # Default 4x upscaling
        
    def __getitem__(self, index):
        filename = self.splits[self.stage][index]
        
        # Load high-resolution target
        hr_image = tifffile.imread(os.path.join(self.hr_path, filename))
        hr_image = torch.from_numpy(hr_image).float()
        hr_image = torch.permute(hr_image, (2, 0, 1))  # CHW format
        
        # Load or create low-resolution input
        lr_path_full = os.path.join(self.lr_path, filename)
        
        if os.path.exists(lr_path_full):
            # Load pre-existing low-res version
            lr_image = tifffile.imread(lr_path_full)
            lr_image = torch.from_numpy(lr_image).float()
            # lr_image = torch.permute(lr_image, (2, 0, 1))
        
        # Apply padding to both
        hr_image = self.pad(hr_image)
        lr_image = self.pad(lr_image)
        
        # Crop to consistent sizes
        hr_image = hr_image[:, 0:192, 0:192]
        # lr_size = 192 // self.scale_factor
        lr_image = lr_image[:, 0:192, 0:192]
        
        return {
            'lr': lr_image,  # Low resolution input
            'hr': hr_image   # High resolution target
        }
    
    def _read_stems_from_csv(self, csv_path):
        if not os.path.isfile(csv_path):
            raise FileNotFoundError('CSV file not found: {}'.format(csv_path))

        stems = []
        with open(csv_path, 'r', newline='') as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row:
                    continue
                value = row[0].strip()
                if not value:
                    continue
                if value.lower() in {'filename', 'file', 'stem', 'id', 'name'}:
                    continue
                if not (value.lower().endswith('.tif') or value.lower().endswith('.tiff')):
                    stems.append(os.path.splitext(os.path.basename(value))[0] + ".tif")
                else:
                    stems.append(os.path.splitext(os.path.basename(value))[0])

        if not stems:
            raise ValueError('No file stems found in first CSV column: {}'.format(csv_path))
        return stems

    def _build_paths_from_stems(self, image_dir, stems):
        all_paths = Util.get_paths_from_images(image_dir)
        stem_to_path = {}
        for path in all_paths:
            stem = os.path.splitext(os.path.basename(path))[0]
            if stem in stem_to_path:
                raise ValueError('Duplicate stem "{}" found in {}'.format(stem, image_dir))
            stem_to_path[stem] = path

        missing = []
        ordered_paths = []
        for stem in stems:
            if stem not in stem_to_path:
                missing.append(stem)
                continue
            ordered_paths.append(stem_to_path[stem])

        if missing:
            preview = ', '.join(missing[:10])
            raise ValueError('Missing {} stems in {}. First missing: {}'.format(len(missing), image_dir, preview))
        return ordered_paths
    
    def __len__(self):
        return len(self.splits[self.stage])#Just for testing


# Modified get_dataloaders function
def get_dataloaders_superres(config, n_workers=4, use_mnist=False):
    """Prepares dataloaders for super-resolution training."""
    
    train_dataset = GeoDatasetSuperRes("train", config)
    val_dataset = GeoDatasetSuperRes("valid", config)

    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=n_workers
    )

    dataloaders['val'] = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=n_workers
    )

    return dataloaders


class GeoDatasetTest(torch.utils.data.Dataset):
    """
    Test/Inference dataset for Super-Resolution evaluation.
    
    Loads matched LR/HR .tif file pairs from separate directories.
    Images are returned as float32 tensors in [-1, 1].
    
    Args:
        lr_dir: Directory containing low-resolution .tif files
        hr_dir: Directory containing high-resolution .tif files
        lr_scale: Scale factor to upsample LR → HR spatial size (bicubic). Set to None to skip.
        normalize: Whether to map from [data_min, data_max] → [-1, 1]
        pad: Padding to apply to images (default: 0)
        crop_size: Target crop size (H, W). Set to None to skip cropping.
    """
    
    def __init__(
        self, 
        lr_path: str, 
        hr_path: str, 
        pad: int = 0,
        crop_size: tuple = None
    ):
        self.lr_path = Path(lr_path)
        self.hr_path = Path(hr_path)
        self.crop_size = crop_size
        self.pad = torch.nn.ZeroPad2d(pad) if pad > 0 else None
        
        # Find all .tif files in both directories
        self.lr_files = sorted(path for path in self.lr_path.iterdir() if path.is_file() and _is_visible_tif(path))
        self.hr_files = sorted(path for path in self.hr_path.iterdir() if path.is_file() and _is_visible_tif(path))
        
        # Match by stem name
        hr_stems = {f.stem: f for f in self.hr_files}
        self.pairs = [(lf, hr_stems[lf.stem]) for lf in self.lr_files if lf.stem in hr_stems]
        
        if len(self.pairs) == 0:
            raise FileNotFoundError(
                f"No matching LR/HR .tif pairs found.\n"
                f"  LR dir: {lr_path}\n"
                f"  HR dir: {hr_path}"
            )
        
        print(f"Found {len(self.pairs)} LR/HR test pairs.")
    
    def __getitem__(self, index):
        lr_path, hr_path = self.pairs[index]  # already matched correctly

        hr_image = tifffile.imread(str(hr_path))
        hr_image = torch.from_numpy(hr_image).float()
        hr_image = torch.permute(hr_image, (2, 0, 1))

        lr_image = tifffile.imread(str(lr_path))
        lr_image = torch.from_numpy(lr_image).float()
        # lr_image = torch.permute(lr_image, (2, 0, 1))

        hr_image = hr_image[:, 0:192, 0:192]
        lr_image = lr_image[:, 0:192, 0:192]

        return {
            'lr': lr_image,
            'hr': hr_image,
            'filename': lr_path.name  # just the bare filename, not full path
        }

    def __len__(self):
        return len(self.pairs)


def get_test_dataloader(lr_path, hr_path, batch_size=1, num_workers=4, pad=0, crop_size=None):
    """
    Create a dataloader for test/inference with LR/HR image pairs.
    
    Args:
        lr_path: Path to low-resolution test images
        hr_path: Path to high-resolution test images
        batch_size: Batch size for dataloader
        num_workers: Number of worker processes
        pad: Padding to apply to images
        crop_size: Target crop size as (H, W) tuple
        
    Returns:
        DataLoader for test dataset
    """
    test_dataset = GeoDatasetTest(
        lr_path=lr_path,
        hr_path=hr_path,
        pad=pad,
        crop_size=crop_size
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return test_loader


