from __future__ import annotations

import os.path
from typing import List

import pytorch_lightning as pl
from lightning_fabric import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from transformers import BertTokenizer

from .dataloader import RegressionDataloader
from .models import BertWithRegression


class RegressionConfig:
    # ----------配置数据------------
    csv_data_path: str
    drop_columns: List[str]
    max_length: int
    tokenizer: BertTokenizer
    batch_size: int = 32
    val_split: float = 0.2
    test_split: float = 0.1
    num_workers: int = 4
    label_set: List[str] | None = None

    # -----------配置模型------------
    vocab_size: int
    nr_frozen_epochs: int
    encoder_learning_rate: float
    fine_tune_learning_rate: float
    model_name: str = "Rostlab/prot_t5_xl_uniref50"
    # Rostlab / prot_t5_xl_uniref50

    # -----------回调配置-----------
    # early stop
    monitor: str
    patience: int = 5
    mode: str = 'min'

    # -----------检查点配置-----------
    checkpoint_save_path: str

    # -----------训练器配置-----------
    accelerator: str = 'auto'
    devices: str | int | List[int] = "auto"
    strategy: str = "auto"
    max_epochs: int
    precision: int
    num_processes: int
    amp_level: str | None
    use_logger: bool = True
    fast_dev_run: bool = False


class Trainer:

    def __init__(self):
        self.conf = RegressionConfig()

    def build_dataloader(self):
        return RegressionDataloader.RegressionDataModule(
            csv_data_path=self.conf.csv_data_path,
            drop_columns=self.conf.drop_columns,
            max_length=self.conf.max_length,
            tokenizer=self.conf.tokenizer,
            batch_size=self.conf.batch_size,
            label_set=self.conf.label_set
        )

    def build_model(self):
        return BertWithRegression.BertWithRegression(
            vocab_size=self.conf.vocab_size,
            nr_frozen_epochs=self.conf.nr_frozen_epochs,
            encoder_learning_rate=self.conf.encoder_learning_rate,
            fine_tune_learning_rate=self.conf.fine_tune_learning_rate,
            model_name=self.conf.model_name
        )

    def build_callback(self):
        early_stop_callback = EarlyStopping(
            monitor=self.conf.monitor,
            min_delta=0.0,
            patience=self.conf.patience,
            verbose=True,
            mode=self.conf.mode,
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(self.conf.checkpoint_save_path, '{epoch}-{val_loss:.2f}'),
            verbose=True,
            monitor=self.conf.monitor,
            mode=self.conf.mode,
        )
        return [early_stop_callback, checkpoint_callback]

    def build_logger(self):
        return

    def train(self):
        seed_everything(42)
        model = self.build_model()
        dl = self.build_dataloader()

        if self.conf.use_logger:
            from pytorch_lightning.loggers import WandbLogger
            wandb_logger = WandbLogger(project="BERT")
            wandb_logger.watch(model)

        trainer = pl.Trainer(
            accelerator=self.conf.accelerator,
            devices=self.conf.devices,
            logger=wandb_logger if self.conf.use_logger else None,
            max_epochs=self.conf.max_epochs,
            callbacks=self.build_callback(),
            precision=self.conf.precision,
            # amp_level=self.conf.amp_level,
            # deterministic=True,
            fast_dev_run=self.conf.fast_dev_run,
        )

        trainer.fit(model, dl)
        trainer.test(model, dl.test_dataloader())
