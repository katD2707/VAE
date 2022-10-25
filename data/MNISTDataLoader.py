import pytorch_lightning as pl
from base import BaseDataModule
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

class MNISTDataModule(BaseDataModule):
    def __init__(self,
                 *args,
                 **kwargs,
                 ):
        super(MNISTDataModule, self).__init__(*args, **kwargs)

        self.transforms = self.data_transforms()

    def data_transforms(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        return transform

    # Download data
    def prepare_data(self, download=True):
        datasets.MNIST(self.data_dir, train=True, download=download)
        datasets.MNIST(self.data_dir, train=False, download=download)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            mnist_full = datasets.MNIST(self.data_dir, train=True, transform=self.transforms)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [len(mnist_full), 0])

        if stage == "test" or stage is None:
            self.mnist_test = datasets.MNIST(self.data_dir, train=False, transform=self.transforms)

    def train_dataloader(self):
        return DataLoader(self.mnist_train,
                          shuffle=True,
                          drop_last=True,
                          **self.dl_dict,
                          )

    def val_dataloader(self):
        return DataLoader(self.mnist_val,
                          shuffle=True,
                          drop_last=True,
                          **self.dl_dict,
                          )

    def test_dataloader(self):
        return DataLoader(self.mnist_test,
                          shuffle=True,
                          drop_last=True,
                          **self.dl_dict,
                          )
