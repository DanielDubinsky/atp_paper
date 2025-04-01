from typing import Callable
from torch.utils.data import Dataset
import cv2
import pandas as pd
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.neighbors import KernelDensity


class ATPDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, root: str, transforms: Callable = ToTensorV2()
    ) -> None:
        self.df = df.copy()
        self.root = root
        self.transforms = transforms
        if "domain_idx" not in self.df.columns:
            self.df["domain_idx"] = self.df.apply(
                lambda x: f"{x.location}_{x.microscope}", axis=1
            )
            domain_map = {d: i for i, d in enumerate(self.df.domain_idx.unique())}
            self.df["domain_idx"] = self.df.domain_idx.map(domain_map)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        im_path = f"{self.root}/{row.location_in_bucket}"
        image = cv2.imread(
            f"{self.root}/{row.location_in_bucket}", cv2.IMREAD_GRAYSCALE
        )
        if image is None:
            raise Exception("Failed to open image", im_path)
        ret = {
            "image": image,
            "name": row.location_in_bucket,
        }
        if "domain_idx" in row:
            ret["domain"] = int(row.domain_idx)

        if "value" in row:
            ret["y_true"] = np.float32(row.value)

        if self.transforms is not None:
            ret = self.transforms(**ret)

        return ret

    def __len__(self) -> int:
        return len(self.df)


class ATPPairDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        root: str,
        transforms: Callable = ToTensorV2(),
        control_sample="fixed",
    ) -> None:
        self.df = df.copy()
        self.root = root
        self.transforms = transforms
        self.control_sample = control_sample

        if "control" not in self.df.columns:
            from atp.preprocess import set_control

            self.df = set_control(self.df)
        if control_sample not in ["random", "fixed"]:
            raise ValueError("control_sample not recognized")

        # sort controls in df
        self.df["control"] = self.df.control.map(
            lambda x: sorted(x, key=lambda y: self.df.loc[y, "value"].item())
        )

    def __getitem__(self, index):
        row = self.df.iloc[index]
        controls = self.df.loc[row.control] # we assume controls are sorted by value
        if self.control_sample == "fixed":
            control = controls.iloc[int(len(controls) // 2)]
        if self.control_sample == "random":
            control = controls.iloc[torch.randint(0, len(controls), size=(1,)).item()]

        image = cv2.imread(f"{self.root}/{row.location_in_bucket}", cv2.IMREAD_GRAYSCALE)
        ref_image = cv2.imread(
            f"{self.root}/{control.location_in_bucket}", cv2.IMREAD_GRAYSCALE
        )
        ret = {
            "image": image,
            "ref": ref_image,
            "name": row.location_in_bucket,
            "y_true": np.float32(row.value / control.value),
        }
        if self.transforms is not None:
            ret = self.transforms(**ret)
            ret["ref"] = self.transforms(image=ref_image)["image"]

        return ret

    def __len__(self) -> int:
        return len(self.df)
