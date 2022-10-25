import pytorch_lightning as pl

class BaseDataModule(pl.LightningDataModule):
    def __init__(self):
        super(BaseDataModule, self).__init__()


    def prepare_data(self):
        raise NotImplementedError

    def setup(self, stage=None):
        raise NotImplementedError

    def train_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError
