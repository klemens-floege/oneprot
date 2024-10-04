from typing import List, Optional, Tuple

import hydra
import pytorch_lightning as L
import pyrootutils
import torch
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger
from omegaconf import DictConfig
import os 
import distributed


print(torch.cuda.is_available())
print(torch.__version__)
print(torch.version.cuda)

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src import utils
from src.utils import task_wrapper, instantiate_callbacks, instantiate_loggers, log_hyperparameters, extras, get_pylogger


log = get_pylogger(__name__)

@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("compile"):
        log.info("Compiling model!")
        model = torch.compile(model)
    
    if cfg.get("ckpt_path"):
        log.info("Loading model weights from checkpoint!")
        checkpoint = torch.load(cfg.ckpt_path)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        log.info("No checkpoint provided, starting training from scratch.")
    
    trainer.fit(model=model, datamodule=datamodule)

    train_metrics = trainer.callback_metrics

    return train_metrics, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        distributed.init_distributed_mode(12354)

    utils.extras(cfg)

    train(cfg)


from pytorch_lightning.plugins.environments import SLURMEnvironment

def patch_lightning_slurm_master_addr():
    if os.getenv('SYSTEMNAME', '') not in [
            'juwelsbooster',
            'juwels',
            'jurecadc',
    ]:
        return

    old_resolver = SLURMEnvironment.resolve_root_node_address

    def new_resolver(self, nodes):
        return old_resolver(nodes) + 'i'

    SLURMEnvironment.__old_resolve_root_node_address = old_resolver
    SLURMEnvironment.resolve_root_node_address = new_resolver

patch_lightning_slurm_master_addr()

if __name__ == "__main__":
    main()