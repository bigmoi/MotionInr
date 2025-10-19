from os import path

import  pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader
from Utils.utils import instantiate_from_config

class DataModule(pl.LightningDataModule):
    def __init__(self,dataset_use,mode='train',batch_size=32,num_workers=4) :
        super().__init__()
        self.save_hyperparameters()
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.dataset_implement=instantiate_from_config(self.hparams.dataset_use)

    def train_dataloader(self):
        return DataLoader(self.dataset_implement, self.hparams.batch_size, num_workers=self.hparams.num_workers)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.dataset_implement, self.hparams.batch_size, num_workers=self.hparams.num_workers)
