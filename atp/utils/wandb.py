import wandb
from hydra.utils import instantiate
import os
import torch
from omegaconf import DictConfig
from sklearn.metrics import r2_score, mean_absolute_error
from atp.regression_module import RegressionModule


def get_run_config(run_path: str) -> DictConfig:
    api = wandb.Api()
    run = api.run(run_path)

    return DictConfig(run.config)


def load_best_model(run_path):
    if os.path.exists(run_path):
        p = run_path
    else:
        workspace, project, id = run_path.split("/")
        api = wandb.Api()
        p = api.artifact(f"{workspace}/{project}/model-{id}:best").download()
        p = os.path.join(p, "model.ckpt")

    return RegressionModule.load_from_checkpoint(
        p, map_location=torch.device("cpu")
    ).eval()
