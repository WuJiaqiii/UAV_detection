"""Install the same deterministic base geometry in YOLO's real training loader.

This is used by train_yolo_frontend.py and the optional weighted equivalence
check. Random augmentation and rectangular batches are disabled for the baseline.
No changes are written inside the user's YOLO submodule.
"""
from pathlib import Path
import cv2
import numpy as np
from util.spectrogram_frontend import SPEC_SHAPE, resize_yolo_image


def install_yolo_dataset_contract():
    import utils.dataloaders as module
    cls = module.LoadImagesAndLabels
    if getattr(cls, "_uav_native_contract", False):
        return module

    original_create = module.create_dataloader

    def load_image(self, index):
        if self.ims[index] is not None:
            return self.ims[index], self.im_hw0[index], self.im_hw[index]
        path = Path(self.npy_files[index])
        image = np.load(path) if path.exists() else cv2.imread(self.im_files[index])
        if image is None:
            raise ValueError("Cannot read " + self.im_files[index])
        if image.shape[:2] != SPEC_SHAPE:
            raise ValueError("YOLO dataset must contain native 512x750 PNGs")
        h0, w0 = image.shape[:2]
        image = resize_yolo_image(image, self.img_size)
        return image, (h0, w0), image.shape[:2]

    def create_dataloader(*args, **kwargs):
        # YOLOv5 passes these by keyword in its training and validation entry.
        kwargs["augment"] = False
        kwargs["rect"] = False
        return original_create(*args, **kwargs)

    cls.load_image = load_image
    cls._uav_native_contract = True
    module.create_dataloader = create_dataloader
    return module
