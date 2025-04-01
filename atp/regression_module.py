import torch
from torch import nn
from hydra.utils import instantiate
from omegaconf import OmegaConf
import lightning as L
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np


def set_norm_layer(config, kwargs):
    if "norm_layer" in config.module.encoder:
        norm_layer = config.module.encoder.norm_layer
        if norm_layer == "instance_norm":
            kwargs.update({"norm_layer": nn.InstanceNorm2d})
        elif norm_layer == "batch_norm":
            kwargs.update({"norm_layer": nn.BatchNorm2d})
        else:
            raise ValueError(f"unknown norm_layer: {norm_layer}")
    return kwargs


def patch_first_conv(model, new_in_channels, default_in_channels=3, pretrained=True):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    # get first conv
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == default_in_channels:
            break

    weight = module.weight.detach()
    module.in_channels = new_in_channels

    if not pretrained:
        module.weight = nn.parameter.Parameter(
            torch.Tensor(
                module.out_channels,
                new_in_channels // module.groups,
                *module.kernel_size,
            )
        )
        module.reset_parameters()

    elif new_in_channels == 1:
        new_weight = weight.sum(1, keepdim=True)
        module.weight = nn.parameter.Parameter(new_weight)
    else:
        raise ValueError()


class RegressionModule(L.LightningModule):
    def __init__(self, config, **kwargs):
        super().__init__()

        self.loss_fn = instantiate(config.module.loss)
        self.config = config

        self._init_encoder(config)
        self._init_head(config)
        self.input_size = torch.Size([1, 1, *config.datamodule.resolution])
        self.save_hyperparameters()

    def _init_encoder(self, config):
        if config.module.encoder.name == "resnet18":
            from torchvision.models import resnet18, ResNet18_Weights

            weights = ResNet18_Weights.DEFAULT
            kwargs = {}
            kwargs = set_norm_layer(config, kwargs)
            self.encoder = resnet18(**kwargs)
            sd = weights.get_state_dict(False)
            if (
                "norm_layer" in config.module.encoder
                and config.module.encoder.norm_layer == "instance_norm"
            ):
                for k in list(sd.keys()):
                    if k.endswith("running_mean") or k.endswith("running_var"):
                        sd.pop(k)
            incompetible_keys = self.encoder.load_state_dict(sd, strict=False)
            if len(incompetible_keys.missing_keys) > 0:
                raise ValueError(
                    f"Falied loading weights, missing keys: {incompetible_keys}"
                )
            self.encoder.fc = nn.Identity()
        elif config.module.encoder.name == "vit_custom":
            from torchvision.models.vision_transformer import _vision_transformer

            params = OmegaConf.to_container(config.module.encoder.params)
            self.encoder = _vision_transformer(**params)
            self.encoder.heads = nn.Identity()
        elif config.module.encoder.name.startswith("torchvision"):
            from hydra.utils import get_method

            torchvision_encoder = get_method(config.module.encoder.name)

            self.encoder = torchvision_encoder(weights=config.module.encoder.weights)
            self.encoder.heads = nn.Identity()
        else:
            raise ValueError(f"Unknown encoder: {config.module.encoder.name}")

        patch_first_conv(self.encoder, 1)
        self.encoder = torch.nn.Sequential(
            self.encoder,
            torch.nn.Flatten(),
        )

        if "freeze" in config.module.encoder and config.module.encoder.freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def _get_latent_dim(self, encoder, input_size) -> torch.Size:
        x = torch.zeros(1, 1, *input_size)

        state = self.training
        self.eval()
        with torch.no_grad():
            size = encoder(x).size().numel()
        self.train(state)
        return size

    def _init_head(self, config):
        latent_dim = self._get_latent_dim(self.encoder, config.datamodule.resolution)

        # Partial instantiation
        self.regression_head = instantiate(config.module.head)
        self.regression_head = self.regression_head(in_channels=latent_dim)
        if config.module.head.hidden_channels[-1] != 1:
            self.regression_head = nn.Sequential(
                self.regression_head,
                nn.Linear(config.module.head.hidden_channels[-1], 1),
            )

    def forward(self, image, **kwargs):
        if type(image) == dict:
            image = image["image"]

        latent = self.encoder(image)
        out = self.regression_head(latent)
        return out

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        assert self.training == False
        y_pred = self.forward(**batch)
        ret = {"y_pred": y_pred}
        if "y_true" in batch:
            ret["y_true"] = batch["y_true"]
        return ret

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def _step(self, batch, stage):
        image = batch["image"]
        y_true = batch["y_true"].view(-1, 1)
        batch_size = image.size(0)

        assert image.ndim == 4

        y_pred = self.forward(**batch)
        loss = self.loss_fn(y_pred, y_true)
        error = y_pred - y_true

        ret = {
            f"loss": loss,
            f"error": error.cpu().detach(),
            "y_pred": y_pred.cpu().detach(),
            "y_true": y_true.cpu().detach(),
        }

        is_training = stage == "train"
        self.log(
            f"{stage}_loss",
            loss,
            on_step=is_training,
            on_epoch=not is_training,
            batch_size=batch_size,
            prog_bar=not is_training,
        )
        self.log(
            f"{stage}_mae",
            error.abs().mean(),
            on_step=is_training,
            on_epoch=not is_training,
            batch_size=batch_size,
            prog_bar=not is_training,
        )

        return ret

    def configure_optimizers(self):
        optimizer = instantiate(self.config.module.optimizer)
        optimizer = optimizer(params=self.parameters())

        lr_scheduler = None
        if (
            "lr_scheduler" in self.config.module
            and self.config.module.lr_scheduler is not None
        ):
            lr_scheduler = instantiate(self.config.module.lr_scheduler)
            lr_scheduler = {
                "scheduler": lr_scheduler(optimizer=optimizer),
                "monitor": self.config.trainer.monitor,
            }
            return [optimizer], [lr_scheduler]
        return [optimizer]


class PairRegressionModule(RegressionModule):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

    def _get_latent_dim(self, encoder, input_size):
        return super()._get_latent_dim(encoder, input_size) * 2

    def forward(self, image, ref, **kwargs):
        latenti = self.encoder(image)
        latentr = self.encoder(ref)
        latent = torch.concat([latenti, latentr], dim=1)
        out = self.regression_head(latent)
        return out
