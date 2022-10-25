import logging
from utils import Conv2dSamePadding
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    '''
        Set up is the same as in 'https://arxiv.org/pdf/1312.6114v10.pdf'
        except encoder and decoder is 2D-Convolution NN instead of MLP
    '''

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: [] = None,
                 **kwargs) -> None:
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
            logging.warning('Default hidden dims is currently used with output channels: %s', hidden_dims)

        self.encoder = Encoder(hidden_dims)

        self.mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.log_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        self.decoder = Decoder(hidden_dims[::-1])

        self.flatten = nn.Flatten()

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, inputs, **kwargs):
        x = self.encoder(inputs)
        x = self.flatten(x)

        # Calculate mu and std
        mu, log_var = self.mu(x), self.log_var(x)

        z = self.reparameterize((mu, log_var))
        z = self.decoder_input(z)
        z = z.view(-1, 512, 2, 2)

        return self.decoder(z), inputs, mu, log_var

    def loss_function(self, *args, **kwargs):
        recons = args[0]
        inputs = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']
        recons_loss = F.mse_loss(recons, inputs)

        kld_loss = torch.mean(-1 / 2. * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss}

    def sample(self,
                 num_samples: int,
                 device):
        z = torch.randn(num_samples,
                        self.latent_dim)
        z = z.to(device)

        samples = self.decode(z)
        return samples

    def generate(self, x, **kwargs):
        return self.forward(x)[0]


class Encoder(nn.Module):
    def __init__(self, hidden_dims):
        super(Encoder, self).__init__()

        layers = []
        # Encoder
        for out_channel in hidden_dims:
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channel,
                          kernel_size=3,
                          stride=2,
                          padding=1,
                          ),
                nn.BatchNorm2d(out_channel),
                nn.LeakyReLU(),
            ))
            in_channels = out_channel

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, hidden_dims):
        super(Decoder, self).__init__()

        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims[-1]),
                    nn.LeakyReLU(),
                )
            )

        self.decoder = nn.Sequential(*layers)

        self.decoder.add_module(nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1,
                               ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1],
                      out_channels=3,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Tanh(),
            )
        )

    def forward(self, z):
        return self.decoder(z)


