import lightning as L
import torch
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
)


class RegressionMetrics(L.Callback):
    """
    After training ends, upload to wandb, best and worst examples of best model on the test test.
    """

    def __init__(self, config):
        super(RegressionMetrics, self).__init__()
        self.config = config
        self._outputs = {}
        for stage in ["train", "val", "test"]:
            self._outputs[stage] = {}
            self._reset_outputs(stage)

    def _reset_outputs(self, stage):
        del self._outputs[stage]
        self._outputs[stage] = {}
        self._outputs[stage]["name"] = []
        self._outputs[stage]["y_pred"] = []
        self._outputs[stage]["y_true"] = []

    def _on_batch_end(self, outputs, batch, stage):
        self._outputs[stage]["name"] += batch["name"]
        self._outputs[stage]["y_pred"] += outputs["y_pred"].flatten().tolist()
        self._outputs[stage]["y_true"] += outputs["y_true"].flatten().tolist()

        if "dis_pred" in outputs:
            if "dis_pred" not in self._outputs[stage]:
                self._outputs[stage]["dis_pred"] = []
                self._outputs[stage]["dis_true"] = []
            self._outputs[stage]["dis_pred"] += outputs["dis_pred"].flatten().tolist()
            self._outputs[stage]["dis_true"] += outputs["dis_true"].flatten().tolist()

    # def on_train_batch_end(
    #     self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs
    # ) -> None:
    #     self._on_batch_end(outputs, batch, "train")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs
    ) -> None:
        if not trainer.sanity_checking:
            self._on_batch_end(outputs, batch, "val")

    def _on_epoch_end(self, stage):
        y_pred = torch.Tensor(self._outputs[stage]["y_pred"])
        y_true = torch.Tensor(self._outputs[stage]["y_true"])

        rsquared = r2_score(y_true=y_true, y_pred=y_pred)
        mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)

        self.log_dict({f"{stage}_rsquared": rsquared, f"{stage}_mae": mae})

        self._reset_outputs(stage)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not trainer.sanity_checking:
            self._on_epoch_end("val")
