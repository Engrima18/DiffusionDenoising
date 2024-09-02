import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from pathlib import Path
import yaml
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


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))  # Optional: Print the configuration for debugging

    wandb.login()
    config = get_config(cfg.generator)
    auto_lr = cfg.auto_lr

    now = datetime.now()
    date_time = now.strftime("%d-%m-%Y-h%H-m%M")
    name = (
        f"{date_time}_{config['loss']}_{config['generation_mode']}_{config['comment']}"
    )

    # W&B options
    logger = pl_loggers.WandbLogger(
        save_dir=config["logging_dir"], project=config["project_name"], name=name
    )

    precision = 16 if config["fp16"] else 32
    accumulate_grad_batches = (
        1
        if not config["accumulate_grad_batches"]
        else config["accumulate_grad_batches"]
    )
    epochs = config["epochs"]
    eval_every = config["eval_every"]

    module = GeneratorModule(config, config["fp16"])
    callback_lr = LearningRateMonitor("step")
    callback_es = EarlyStopping(monitor="val/loss", patience=10)
    callback_last_ckpt = ModelCheckpoint(
        every_n_epochs=1, filename="last_{epoch}_{step}"
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

    checkpoint_path = Path(config["checkpoint"] + name)
    finetune_path = checkpoint_path if config["fine_tune_from"] else None
    trainer.fit(module, ckpt_path=finetune_path, default_root_dir=checkpoint_path)

    # save config to file
    save_path = Path(logger.experiment.get_logdir()) / Path("config.yaml")
    with open(save_path, "w") as f:
        yaml.dump(OmegaConf.to_container(cfg, resolve=True), f)

    wandb.finish()


if __name__ == "__main__":
    main()
