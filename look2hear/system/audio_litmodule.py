###
# Author: Kai Li
# Date: 2022-05-26 18:09:54
# Email: lk21@mails.tsinghua.edu.cn
# LastEditTime: 2024-01-24 00:00:28
###
import gc
from omegaconf import OmegaConf
import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections.abc import MutableMapping
from omegaconf import ListConfig

def flatten_dict(d, parent_key="", sep="_"):
    """Flattens a dictionary into a single-level dictionary while preserving
    parent keys. Taken from
    `SO <https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys>`_

    Args:
        d (MutableMapping): Dictionary to be flattened.
        parent_key (str): String to use as a prefix to all subsequent keys.
        sep (str): String to use as a separator between two key levels.

    Returns:
        dict: Single-level dictionary, flattened.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class AudioLightningModule(pl.LightningModule):
    def __init__(
        self,
        model=None,
        discriminator=None,
        optimizer=None,
        loss_func=None,
        metrics=None,
        scheduler=None,
        grad_accum=1,
    ):
        super().__init__()
        self.audio_model = model
        self.discriminator = discriminator
        self.optimizer = list(optimizer)
        self.loss_func = loss_func
        self.metrics = metrics
        self.scheduler = list(scheduler)
        
        # Save lightning"s AttributeDict under self.hparams
        self.default_monitor = "val_loss"
        # self.print(self.audio_model)
        self.validation_step_outputs = []
        self.test_step_outputs = []
        # emulate a larger effective batch on a single GPU (paper: 8 x batch 1)
        self.grad_accum = max(1, int(grad_accum))
        self._micro_step = 0
        self.automatic_optimization = False

    def forward(self, wav):
        """Applies forward pass of the model.

        Returns:
            :class:`torch.Tensor`
        """
        return self.audio_model(wav)

    def training_step(self, batch, batch_nb):
        ori_data, codec_data = batch
        optimizer_g, optimizer_d = self.optimizers()
        # multiple schedulers
        scheduler_g, scheduler_d = self.lr_schedulers()

        accum = self.grad_accum
        micro = self._micro_step
        self._micro_step = (micro + 1) % accum
        first = micro == 0
        last = micro == accum - 1

        # ---- discriminator update (accumulated) ----
        output = self(codec_data)
        if first:
            optimizer_d.zero_grad(set_to_none=True)
        est_outputs, _ = self.discriminator(output.detach(), sample_rate=44100)
        target_outputs, _ = self.discriminator(ori_data, sample_rate=44100)
        loss_d, d_stats = self.loss_func["d"](target_outputs, est_outputs, return_stats=True)
        self.manual_backward(loss_d / accum)
        if last:
            self.clip_gradients(optimizer_d, gradient_clip_val=5, gradient_clip_algorithm="norm")
            optimizer_d.step()

        # ---- generator update (accumulated) ----
        est_outputs, est_feature_maps = self.discriminator(output, sample_rate=44100)
        _, targets_feature_maps = self.discriminator(ori_data, sample_rate=44100)
        loss_g, g_stats = self.loss_func["g"](
            est_outputs, est_feature_maps, targets_feature_maps, output, ori_data, return_stats=True
        )
        if first:
            optimizer_g.zero_grad(set_to_none=True)
        self.manual_backward(loss_g / accum)
        if last:
            self.clip_gradients(optimizer_g, gradient_clip_val=5, gradient_clip_algorithm="norm")
            optimizer_g.step()

        if self.trainer.is_last_batch:
            scheduler_g.step()
            scheduler_d.step()

        self.log(
            "train_loss_d",
            loss_d,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )

        self.log(
            "train_loss_g",
            loss_g,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )

        # component breakdown: makes a stalled GAN diagnosable at epoch level
        for name, value in g_stats.items():
            self.log(f"train_G_{name}", value, on_epoch=True, sync_dist=True, logger=True)
        for name, value in d_stats.items():
            self.log(f"train_D_{name}", value, on_epoch=True, sync_dist=True, logger=True)

        return loss_g

    def on_train_epoch_end(self):
        """Print the GAN component breakdown once per epoch (visible in .out log)."""
        cm = self.trainer.callback_metrics
        keys = [
            "train_loss_g", "train_G_rec", "train_G_fm", "train_G_adv",
            "train_loss_d", "train_D_real", "train_D_fake",
        ]
        parts = [f"{k}={float(cm[k]):.4f}" for k in keys if k in cm]
        if parts:
            print(f"[epoch {self.current_epoch}] " + "  ".join(parts), flush=True)

    def validation_step(self, batch, batch_nb):
        # cal val loss
        # MusdbMoisesdbEval returns (ori_wav, codec_wav, path)
        ori_data, codec_data, _ = batch
        # print(mixtures.shape)
        est_sources = self(codec_data)
        loss = self.metrics(est_sources, ori_data)
        
        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )
        
        self.validation_step_outputs.append(loss)
        
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        # val
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        val_loss = torch.mean(self.all_gather(avg_loss))
        self.log(
            "lr",
            self.optimizer[0].param_groups[0]["lr"],
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.logger.experiment.log(
            {"learning_rate": self.optimizer[0].param_groups[0]["lr"], "epoch": self.current_epoch}
        )
        self.logger.experiment.log(
            {"val_pit_sisnr": -val_loss, "epoch": self.current_epoch}
        )

        self.validation_step_outputs.clear()  # free memory
        torch.cuda.empty_cache()
        
    def test_step(self, batch, batch_nb):
        # MusdbMoisesdbEval returns (ori_wav, codec_wav, path)
        mixtures, targets, _ = batch
        est_sources = self(mixtures)
        loss = self.metrics(est_sources, targets)
        self.log(
            "test_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )
        self.test_step_outputs.append(loss)
        return {"test_loss": loss}
    
    def on_test_epoch_end(self):
        # val
        avg_loss = torch.stack(self.test_step_outputs).mean()
        test_loss = torch.mean(self.all_gather(avg_loss))
        self.log(
            "lr",
            self.optimizer.param_groups[0]["lr"],
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.logger.experiment.log(
            {"learning_rate": self.optimizer.param_groups[0]["lr"], "epoch": self.current_epoch}
        )
        self.logger.experiment.log(
            {"test_pit_sisnr": -test_loss, "epoch": self.current_epoch}
        )

        self.test_step_outputs.clear()

    def configure_optimizers(self):
        """Initialize optimizers, batch-wise and epoch-wise schedulers."""
        if self.scheduler is None:
            return self.optimizer
        if not isinstance(self.scheduler, (list, tuple)):
            self.scheduler = [self.scheduler]  # support multiple schedulers
            
        if not isinstance(self.optimizer, (list, tuple)):
            self.optimizer = [self.optimizer]  # support multiple schedulers
        
        epoch_schedulers = []
        for sched in self.scheduler:
            if not isinstance(sched, dict):
                if isinstance(sched, ReduceLROnPlateau):
                    sched = {"scheduler": sched, "monitor": self.default_monitor}
                epoch_schedulers.append(sched)
            else:
                sched.setdefault("monitor", self.default_monitor)
                sched.setdefault("frequency", 1)
                # Backward compat
                if sched["interval"] == "batch":
                    sched["interval"] = "step"
                assert sched["interval"] in [
                    "epoch",
                    "step",
                ], "Scheduler interval should be either step or epoch"
                epoch_schedulers.append(sched)
        return self.optimizer, epoch_schedulers
            
    @staticmethod
    def config_to_hparams(dic):
        """Sanitizes the config dict to be handled correctly by torch
        SummaryWriter. It flatten the config dict, converts ``None`` to
        ``"None"`` and any list and tuple into torch.Tensors.

        Args:
            dic (dict): Dictionary to be transformed.

        Returns:
            dict: Transformed dictionary.
        """
        dic = flatten_dict(dic)
        for k, v in dic.items():
            if v is None:
                dic[k] = str(v)
            elif isinstance(v, (list, tuple)):
                dic[k] = torch.tensor(v)
        return dic
