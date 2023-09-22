from pathlib import Path

import numpy as np
import cv2
from scipy.ndimage.interpolation import rotate

import torch

class DetLoaderPatch(object):
    def __init__(self, base_path):
        self.img_paths = sorted(Path(f"{base_path}/in").glob("*.tif"))
        self.gt_paths = sorted(Path(f"{base_path}/gt").glob("*.tif"))

        self.crop_size = (256, 256)
    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, data_id):
        img_path = self.img_paths[data_id]
        img = cv2.imread(str(img_path))
        img = img / 255

        gt_path = self.gt_paths[data_id]
        gt = cv2.imread(str(gt_path), 0)
        gt = gt / 255

        rand_value = np.random.randint(0, 4)
        img = rotate(img, 90 * rand_value, mode="nearest")
        gt = rotate(gt, 90 * rand_value, mode="nearest")

        img = torch.from_numpy(img.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))

        return img.permute(2, 0, 1), gt.unsqueeze(0)


class DetectionTest(object):
    def __init__(self, img_dirs):
        self.img_paths = []
        img_dirs = Path(img_dirs).iterdir()
        for img_dir in img_dirs:
            img_paths = sorted(img_dir.glob("*.png"))
            self.img_paths.extend(img_paths)
    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(str(img_path))
        img = img / 255

        img = torch.from_numpy(img.astype(np.float32))
        return img.permute(2, 0, 1), idx





