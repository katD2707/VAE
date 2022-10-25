import pytorch_lightning as pl

class BaseDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir='./data',
                 batch_size=128,
                 num_workers=1,
                 ):
        super(BaseException, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dl_dict = {'batch_size': self.batch_size,
                        'num_workers': self.num_workers,
                        }

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
