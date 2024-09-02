import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from pathlib import Path
import torch as th
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
import wandb
from src.trainer import GeneratorModule
from src.utils import get_config


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:

    th.set_float32_matmul_precision("high")
    print(OmegaConf.to_yaml(cfg))
    wandb.login()
    auto_lr = cfg.trainer.auto_lr

    now = datetime.now()
    date_time = now.strftime("%d-%m-%Y-h%H-m%M")
    name = f"{date_time}_{cfg.model.loss}_{cfg.comment}"

    # W&B options
    logger = pl_loggers.WandbLogger(
        save_dir=cfg.trainer.logging_dir, project=cfg.project_name, name=name
    )

    precision = 16 if cfg.trainerfp16 else 32
    accumulate_grad_batches = (
        1 if not cfg.trainer.accumulate_grad_batches else cfg.accumulate_grad_batches
    )
    epochs = cfg.trainer.epochs
    eval_every = cfg.trainer.eval_ever
    checkpoint_path = Path(cfg.trainer.checkpoint + name)

    module = GeneratorModule(cfg, cfg.trainer.fp16)
    callback_lr = LearningRateMonitor("step")
    callback_es = EarlyStopping(monitor="val/loss", patience=10)
    callback_last_ckpt = ModelCheckpoint(
        dirpath=checkpoint_path, every_n_epochs=1, filename="last_{epoch}_{step}"
    )
    callback_best_ckpt = ModelCheckpoint(
        every_n_epochs=1,
        filename="best_{epoch}_{step}",
        monitor="val/weighted_loss",
        mode="min",
    )

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[callback_lr, callback_es, callback_last_ckpt, callback_best_ckpt],
        accelerator="gpu",
        max_epochs=epochs,
        check_val_every_n_epoch=eval_every,
        precision=precision,
        accumulate_grad_batches=accumulate_grad_batches,
    )

    if auto_lr:
        lr_finder = trainer.tuner.lr_find(
            module,
            min_lr=1e-5,
            max_lr=1e-1,
        )
        lr = lr_finder.suggestion["lr"]
        print(f"Suggested learning rate: {lr}")
        module.hparams.lr = lr

    finetune_path = checkpoint_path if cfg["fine_tune_from"] else None
    trainer.fit(module, ckpt_path=finetune_path)

    # save config to file
    # save_path = Path(logger.experiment.get_logdir()) / Path("config.yaml")
    # with open(save_path, "w") as f:
    #     yaml.dump(OmegaConf.to_container(cfg, resolve=True), f)

    wandb.finish()


if __name__ == "__main__":
    main()
