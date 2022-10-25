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
            full = datasets.MNIST(self.data_dir, train=True, transform=self.transforms)
            self.train, self.val = random_split(full, [len(full), 0])

        if stage == "test" or stage is None:
            self.test = datasets.MNIST(self.data_dir, train=False, transform=self.transforms)

    def train_dataloader(self):
        return DataLoader(self.train,
                          shuffle=True,
                          drop_last=True,
                          **self.dl_dict,
                          )

    def val_dataloader(self):
        return DataLoader(self.val,
                          shuffle=True,
                          drop_last=True,
                          **self.dl_dict,
                          )

    def test_dataloader(self):
        return DataLoader(self.test,
                          shuffle=True,
                          drop_last=True,
                          **self.dl_dict,
                          )
