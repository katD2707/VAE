import argparse
import yaml
import numpy as np
from torch.backends import cudnn
import torch
from models.model_list import VAE_MODELS
from data import data_list
from experiment import VAEXperiment
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger




def train(config):
    # Load model
    model = VAE_MODELS[config['model_params']['name']](**config['model_params'])
    experiment = VAEXperiment(model, config['exp_params'])
    dataloader = data_list.DATASET[config['exp_params']['dataset']]
    tt_logger = CSVLogger(
        save_dir=config['logging_params']['save_dir'],
        name=config['logging_params']['name'],
    )
    runner = Trainer(
        min_epochs=1,
        logger=tt_logger,
        log_every_n_steps=100,
        num_sanity_val_steps=5,
        **config['trainer_params']
    )

    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment, dataloader)

if __name__ == "__main__":
    # Parse parameters
    parser = argparse.ArgumentParser(description='Trainer for VAE')
    parser.add_argument('--params', '-p',
                        help='path to the config file',
                        default='configs/vae.yaml',
                        required=True,
                        type=str,
                        )
    args = parser.parse_args()
    with open(args.params, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    # For reproducibility
    torch.manual_seed(config['logging_params']['manual_seed'])
    np.random.seed(config['logging_params']['manual_seed'])
    cudnn.deterministic = True
    cudnn.benchmark = False

    # Call the training function
    train(config)
