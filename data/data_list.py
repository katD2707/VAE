from .MNISTDataLoader import MNISTDataModule
from .CelebADataLoader import CelebADataModule

DATASET = {
    'mnist': MNISTDataModule,
    'celeba': CelebADataModule,
}