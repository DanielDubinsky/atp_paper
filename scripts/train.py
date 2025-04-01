import hydra
import lightning as L
from omegaconf import DictConfig
from lightning.pytorch.loggers import TensorBoardLogger
from hydra.utils import instantiate
from atp.callbacks import RegressionMetrics
import random


def finalize(lightningmodule, datamodule, trainer):
    import torch
    from sklearn.metrics import r2_score, mean_absolute_error
    import re
    import os
    import pandas as pd
    import yaml

    best_model_path = trainer.checkpoint_callback.best_model_path
    best_epoch = re.match(r".*epoch=(\d+).*", best_model_path).groups()[0]
    os.symlink(
        os.path.basename(best_model_path),
        f"{lightningmodule.logger.log_dir}/checkpoints/best.ckpt",
    )

    preds = trainer.predict(lightningmodule, datamodule=datamodule, ckpt_path="best")
    y_true = torch.concat([p["y_true"] for p in preds]).ravel()
    y_pred = torch.concat([p["y_pred"] for p in preds]).ravel()
    df = datamodule.predict_dataset.df.copy()
    df["y_true"] = y_true
    df["y_pred"] = y_pred

    best_dict = {}
    best_dict["epoch"] = best_epoch
    for idx, group in df.groupby(["domain", "split"]):
        loc, split = idx
        best_dict[f"{split}_{loc}_r2"] = r2_score(group.y_true, group.y_pred).item()
        best_dict[f"{split}_{loc}_mae"] = mean_absolute_error(
            group.y_true, group.y_pred
        ).item()

    with open(f"{lightningmodule.logger.log_dir}/best_metrics.yaml", "w") as f:
        yaml.dump(best_dict, f)

    df.to_csv(f"{lightningmodule.logger.log_dir}/data.csv", index=False)


@hydra.main(config_path="config", config_name="0001.yaml", version_base="1.3")
def main(config: DictConfig):
    if config.name is None:
        config.name = "".join(
            random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=6)
        )
    loggers = [TensorBoardLogger(save_dir=config.log_dir, name=config.name)]

    L.seed_everything(seed=config.trainer.seed, workers=True)
    lightningmodule = instantiate(config.module.cls)
    lightningmodule = lightningmodule(config)
    datamodule = instantiate(config.datamodule)
    callbacks = list(instantiate(config.callbacks).values()) + [
        RegressionMetrics(config)
    ]

    trainer = L.Trainer(
        max_epochs=int(config.trainer.max_epochs),
        logger=loggers,
        log_every_n_steps=10,
        callbacks=callbacks,
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        deterministic=True,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        num_sanity_val_steps=0,
    )

    trainer.fit(lightningmodule, datamodule=datamodule)
    finalize(lightningmodule, datamodule, trainer)


if __name__ == "__main__":
    main()
