import click
import lightning as L
import pandas as pd
from atp.preprocess import NormByCompound
from atp.dataset import ATPDataset
from atp.regression_module import RegressionModule
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import r2_score, mean_absolute_error


@click.command()
@click.argument("model_path")
@click.argument("df_path")
@click.argument("root")
def evaluate(model_path, df_path, root):
    df = pd.read_csv(df_path).query('split == "test"')
    df = NormByCompound()(df)

    model = RegressionModule.load_from_checkpoint(model_path, map_location="cpu").eval()

    transforms = A.Compose(
        [
            A.Resize(*model.config["datamodule"]["resolution"], interpolation=5),
            A.Normalize(0.5, 0.5),
            ToTensorV2(),
        ]
    )
    ds = ATPDataset(df, root, transforms)
    dl = DataLoader(ds, batch_size=8)

    trainer = L.Trainer()
    preds = trainer.predict(model, dl)
    y_true = torch.concat([p["y_true"] for p in preds])
    y_pred = torch.concat([p["y_pred"] for p in preds])

    df["y_true"] = y_true
    df["y_pred"] = y_pred

    print("r2")
    print(r2_score(y_true, y_pred))
    if "domain" in df.columns:
        print(df.groupby("domain").apply(lambda x: r2_score(x.y_true, x.y_pred)))
    print("mae")
    print(mean_absolute_error(y_true, y_pred))
    if "domain" in df.columns:
        print(
            df.groupby("domain").apply(
                lambda x: mean_absolute_error(x.y_true, x.y_pred)
            )
        )


if __name__ == "__main__":
    evaluate()
