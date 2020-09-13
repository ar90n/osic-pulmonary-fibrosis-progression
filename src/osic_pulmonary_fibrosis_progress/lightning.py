import gc
import os
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.core import LightningModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.metrics.classification import AUROC
from pytorch_lightning.utilities import rank_zero_info
from torch import optim

from .config import Config
from .util import clean_up, get_device, is_tpu_available, to_device

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
except ImportError:
    pass


def label_smoothing(y: torch.tensor, alpha: float) -> torch.tensor:
    return y.float() * (1 - alpha) + 0.5 * alpha


class LightningModelBase(LightningModule):
    def __init__(self, config: Config, **kwargs):
        super().__init__()
        config_dict = asdict(config) if isinstance(config, Config) else config
        self.save_hyperparameters(config_dict)
        self.test_results = []

    def loss(self, y_hat, y):
        y_smo = label_smoothing(y, self.hparams.label_smoothing)
        return F.binary_cross_entropy_with_logits(
            y_hat,
            y_smo.type_as(y_hat),
            pos_weight=torch.tensor(self.hparams.pos_weight),
        )

    def step(self, batch):
        # return batch loss
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return loss, y, y_hat.sigmoid()

    def training_step(self, batch, batch_idx):
        # hardware agnostic training
        loss, y, y_hat = self.step(batch)
        acc = (y_hat.round() == y).float().mean()
        return {"loss": loss, "acc": acc}

    def training_step_end(self, outputs):
        avg_loss = outputs["loss"].mean()
        avg_acc = outputs["acc"].mean()

        metrics = {"loss": avg_loss, "acc": avg_acc}
        self.logger.log_metrics(metrics, step=self.global_step)
        return metrics

    def validation_step(self, batch, batch_idx):
        loss, y, y_hat = self.step(batch)
        return {"val_loss": loss, "y_hat": y_hat, "y": y}

    def validation_step_end(self, outputs):
        loss = outputs["val_loss"].mean()
        y_hat = outputs["y_hat"]
        y = outputs["y"]
        return {"val_loss": loss, "y_hat": y_hat, "y": y}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        y = torch.cat([x["y"] for x in outputs], dim=0).flatten()
        y_hat = torch.cat([x["y_hat"] for x in outputs], dim=0).flatten()

        avg_acc = (y_hat.round() == y).float().mean()

        auc = AUROC()(y_hat, y.float()) if y.float().mean() > 0 else torch.tensor(0.5)
        metrics = {"val_loss": avg_loss, "val_acc": avg_acc, "val_auc": auc}

        self.logger.log_metrics(metrics, step=self.current_epoch)
        return metrics

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10, eta_min=0.001
        )
        return [optimizer], [scheduler]


#    def setup(self, stage):
#        if is_tpu_available() and isinstance(self.logger, WandbLogger):
#            self.logger._name = self.logger.experiment.name
#            self.logger._offset = True
#            self.logger._experiment = None


class WorkaroundEarlyStopping(EarlyStopping):
    def _stop_distributed_training(self, trainer, pl_module):
        if trainer.use_ddp or trainer.use_ddp2:
            stop = torch.tensor(int(trainer.should_stop), device=pl_module.device)
            dist.all_reduce(stop, op=dist.reduce_op.SUM)
            dist.barrier()
            trainer.should_stop = stop == trainer.world_size

        if trainer.use_tpu:
            stop = torch.tensor(
                int(trainer.should_stop), device=pl_module.device, dtype=torch.int32
            )
            stop = xm.mesh_reduce("stop_signal", stop, lambda xs: torch.stack(xs).sum())
            torch_xla.core.xla_model.rendezvous(
                "pl.EarlyStoppingCallback.stop_distributed_training_check"
            )
            trainer.should_stop = int(stop.item()) == trainer.world_size


class Trainer(Trainer):
    def __init__(
        self,
        config: Config,
        fold_index: Optional[int] = None,
        n_fold: Optional[int] = None,
        **kwargs,
    ):
        if torch.cuda.is_available():
            kwargs["gpus"] = config.gpus
            kwargs["precision"] = config.precision if False else 32

        if is_tpu_available():
            tpu_id = (fold_index + 1) if fold_index is not None else 1
            kwargs["tpu_cores"] = [tpu_id] if config.tpus == 1 else config.tpus
            kwargs["precision"] = config.precision

        if "checkpoint_callback" not in kwargs:
            fold_fmt = None
            if fold_index is not None:
                fold_fmt = f"fold={fold_index}"
                if n_fold is not None:
                    fold_fmt = f"{fold_fmt}@{n_fold - 1}"

            ckpt_filepath_fmt = "{epoch}-{val_auc:.2f}"
            if fold_fmt is not None:
                ckpt_filepath_fmt = f"{fold_fmt}_{ckpt_filepath_fmt}"

            kwargs["checkpoint_callback"] = ModelCheckpoint(
                filepath=str(Path.cwd() / ckpt_filepath_fmt),
                verbose=True,
                monitor="val_auc",
                mode="max",
            )
        if "early_stop_callback" not in kwargs:
            kwargs["early_stop_callback"] = WorkaroundEarlyStopping(
                patience=config.early_stop_patience,
                verbose=True,
                monitor="val_auc",
                mode="max",
            )

        super().__init__(max_epochs=config.max_epochs, **kwargs)
        self._checkpoint_callback = kwargs["checkpoint_callback"]

    @property
    def best_model_path(self):
        return self._checkpoint_callback.best_model_path


class Classifier:
    def __init__(self, model, tta_epochs=1, device=None, with_features=False):
        self.tta_epochs = tta_epochs
        self.device = get_device() if device is None else device
        self.model = model.eval().to(self.device)
        self._with_features = with_features

    def predict(self, data_loader):
        clean_up()

        all_predicts = []
        all_features = []
        with torch.no_grad():
            for i in range(self.tta_epochs):
                rank_zero_info(f"Start {i}-th tta")

                pred, feature = self._predict_once(data_loader)
                all_predicts.append(pred)
                all_features.append(feature)

        avg_predicts = torch.stack(all_predicts).mean(0).cpu().numpy()
        stack_features = torch.stack(all_features).cpu().numpy()

        if self._with_features:
            return avg_predicts, stack_features
        return avg_predicts

    def _predict_once(self, data_loader):
        agg_device = (
            self.device if self.device.type.startswith("cuda") else torch.device("cpu")
        )

        features = []
        results = []
        for x in data_loader:
            x = to_device(x)
            feature = self.model.features(x)
            logit = self.model.classify(feature)
            result = torch.sigmoid(logit)
            features.append(feature.to(agg_device))
            results.append(result.to(agg_device))
        return torch.cat(results), torch.cat(features)
