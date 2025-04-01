from typing import Tuple, Callable, List
import lightning as L
from torch.utils.data import DataLoader
from atp.dataset import ATPDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
from hydra.utils import get_class


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        dataframe_path: str,
        root: str,
        resolution: Tuple[int, int],
        preprocessors: List[Callable],
        batch_size: int,
        num_workers: int,
        train_transforms,
        splitter,
        dataset_class=ATPDataset,
    ) -> None:
        super().__init__()
        self.root = root
        self.dataframe_path = dataframe_path
        self.resolution = resolution
        self.preprocessors = preprocessors
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.splitter = splitter
        self.train_transforms = train_transforms
        self.dataset_class = dataset_class

    def setup(self, stage) -> None:
        # Load dataframe into datasets
        self.df = (
            pd.read_csv(self.dataframe_path)
            if type(self.dataframe_path) == str
            else self.dataframe_path
        )
        for p in self.preprocessors:
            self.df = p(self.df)

        # Transforms
        self.test_transforms = A.Compose(
            [
                A.Resize(*self.resolution, interpolation=5),
                A.Normalize(0.5, 0.5),
                ToTensorV2(),
            ],
        )
        self.train_transforms = A.Compose(
            [
                A.Resize(*self.resolution, interpolation=5),
                self.train_transforms,
                A.Normalize(0.5, 0.5),
                ToTensorV2(),
            ]
        )
        if self.splitter is not None:
            self.df = self.splitter(self.df)
        self.train_dataset = self.dataset_class(
            df=self.df.query('split == "train"'),
            root=self.root,
            transforms=self.train_transforms,
        )
        self.val_dataset = self.dataset_class(
            df=self.df.query('split == "val"'),
            root=self.root,
            transforms=self.test_transforms,
        )
        self.test_dataset = self.dataset_class(
            df=self.df.query('split == "test"'),
            root=self.root,
            transforms=self.test_transforms,
        )
        self.predict_dataset = self.dataset_class(
            df=self.df,
            root=self.root,
            transforms=self.test_transforms,
        )

        self.dataloader_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": True,
            "persistent_workers": self.num_workers > 0,
        }

        # self.example_input_array = (
        #     next(iter(self.train_dataloader()))["image"].cpu().detach()
        # )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset, shuffle=True, **self.dataloader_kwargs
        )

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, **self.dataloader_kwargs)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset, **self.dataloader_kwargs)

    def predict_dataloader(self):
        return DataLoader(dataset=self.predict_dataset, **self.dataloader_kwargs)
