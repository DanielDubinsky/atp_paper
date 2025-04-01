import click
import lightning as L
import pandas as pd
from atp.dataset import ATPDataset
from atp.regression_module import RegressionModule
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import torch


@click.command()
@click.argument("model_path")
@click.argument("df_path")
@click.argument("root")
@click.argument("output")
def predict(model_path, df_path, root, output):
    df = pd.read_csv(df_path)

    model = RegressionModule.load_from_checkpoint(model_path, map_location="cpu").eval()

    transforms = A.Compose(
        [
            A.Resize(*model.config["datamodule"]["resolution"], interpolation=5),
            A.Normalize(0.5, 0.5),
            ToTensorV2(),
        ]
    )
    ds = ATPDataset(df, root, transforms)
    dl = DataLoader(
        ds, batch_size=8, pin_memory=True, num_workers=8, persistent_workers=True
    )

    trainer = L.Trainer(devices=[0])
    preds = trainer.predict(model, dl)
    y_pred = torch.concat([p["y_pred"] for p in preds])

    df["y_pred"] = y_pred
    df.to_csv(output, index=False)


if __name__ == "__main__":
    predict()
