# This file is a part of atari-active-learning project.
# Copyright (c) 2022 Benjamin Ayton (aytonb@mit.edu), Masataro Asai (guicho2.71828@gmail.com, masataro.asai@ibm.com), IBM Corporation

# Make files in VAE-IW importable without changing source code
import sys
sys.path.insert(0, './VAE-IW')
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli, kl_divergence
import torchvision
import numpy as np
import math
#from training import loss_function
from vae.models import ResidualBlock, Dif_ResidualBlock, Res_3d_Conv_15
from vae.utils import linear_schedule, CropImage

from printing_util import number_has_leading_zeros_p

# This is a copy from VAE-IW/training.py, because we cannot import directly
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, qy, z, epoch, choose_loss = "MSE", beta=0.0001, sigma=1,
                  image_size=(128, 128)):
    image_pixels = image_size[0] * image_size[1]
    # Loss from article on SAE
    if choose_loss == "BCE_zerosup":
        # Binary cross entropy loss (negative log likelihood)
        BCE = F.binary_cross_entropy(recon_x.view(-1,image_pixels), x.view(-1, image_pixels),reduction='sum') / x.shape[0]

        pz = Bernoulli(probs=torch.tensor(.5))
        KLD = kl_divergence(qy, pz).sum((1,2,3)).mean()

        # Tries to avoid constant flipping of coins
        zerosuppress_loss = torch.mean(z) * linear_schedule(0.7, epoch)
        return BCE + KLD*beta + zerosuppress_loss, BCE, KLD, F.mse_loss(recon_x,x)

    elif choose_loss == "BCE_low_prior":

        # Binary cross entropy loss (negative log likelihood)
        BCE = F.binary_cross_entropy(recon_x.view(-1, image_pixels), x.view(-1, image_pixels), reduction='sum') / \
              x.shape[0]

        pz = Bernoulli(probs=torch.tensor(.1)) # prior distribution near 0
        KLD = kl_divergence(qy, pz).sum((1, 2, 3)).mean()

        return BCE + KLD*beta, BCE, KLD, F.mse_loss(recon_x, x)

    elif choose_loss == "BCE_std_prior":

        # Binary cross entropy loss (negative log likelihood)
        BCE = F.binary_cross_entropy(recon_x.view(-1, image_pixels), x.view(-1, image_pixels), reduction='sum') / \
              x.shape[0]

        pz = Bernoulli(probs=torch.tensor(.5))
        KLD = kl_divergence(qy, pz).sum((1, 2, 3)).mean()

        return BCE + KLD*beta, BCE, KLD, F.mse_loss(recon_x, x)

    elif choose_loss == "MSE_zerosup":
        # KL-divergence
        pz = Bernoulli(probs=torch.tensor(.5))
        KLD = kl_divergence(qy, pz).sum((1,2,3)).mean()

        # Gaussian log likelihood
        pxz = Normal(recon_x, sigma)
        log_likelihood = -pxz.log_prob(x).sum((1,2,3)).mean()

        # Zero suppress
        zerosuppress_loss = torch.mean(z) * linear_schedule(0.7, epoch)
        return log_likelihood + KLD*beta, log_likelihood, KLD, F.mse_loss(recon_x,x)

    elif choose_loss == "MSE_low_prior":
        # KL-divergence
        pz = Bernoulli(probs=torch.tensor(.1)) # prior distribution near 0
        KLD = kl_divergence(qy, pz).sum((1,2,3)).mean()

        # Gaussian log likelihood
        pxz = Normal(recon_x, sigma)
        log_likelihood = -pxz.log_prob(x).sum((1,2,3)).mean()

        return log_likelihood + KLD*beta, log_likelihood, KLD, F.mse_loss(recon_x,x)

    elif choose_loss == "MSE_std_prior":
        # KL-divergence
        pz = Bernoulli(probs=torch.tensor(.5))
        KLD = kl_divergence(qy, pz).sum((1,2,3)).mean()

        # Gaussian log likelihood
        pxz = Normal(recon_x, sigma)
        log_likelihood = -pxz.log_prob(x).sum((1,2,3)).mean()

        return log_likelihood + KLD*beta, log_likelihood, KLD, F.mse_loss(recon_x,x)


def make_model(zdim=20, image_training_size=128, image_channels=1, loss="BCE"):
    """Makes a VAE model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # The temperature parameter of Res_3D_Conv_15 is not used, instead relying on a separate input so set it to None
    # freely here.
    if loss == "MSE":
        model = No_Activation_Res_3d_Conv_15(zdim,
                                             ResidualBlock,
                                             image_training_size,
                                             None,
                                             image_channels).to(device)
    else:
        model = Res_3d_Conv_15(zdim,
                               ResidualBlock,
                               image_training_size,
                               None,
                               image_channels).to(device)
    return model


def train_epoch(model, optimizer, train_loader, epoch, max_temperature=1.0, min_temperature=0.5, hard=False,
                sigma=0.1, beta=1.0, loss_fn="BCE", log_interval=10, writer=None, writer_epoch=None):
    min_temperature = 0.5
    # Anneal from temperature down to min_temperature in 90 epochs
    ANNEAL_RATE = math.log(max_temperature / min_temperature) / 90

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    train_loss = 0
    train_loss_true = 0
    kld_loss = 0
    reconstruction_loss = 0
    temperature = np.maximum(max_temperature * np.exp(-ANNEAL_RATE * epoch), min_temperature)

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)

        optimizer.zero_grad()
        recon_batch, qz, z = model(data, temperature, hard)

        loss, recons, kld, mse = loss_function(recon_batch, data, qz, z, epoch, choose_loss=loss_fn, sigma=sigma,
                                               beta=beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        train_loss += loss.item() * len(data)
        train_loss_true += (recons.item() + kld.item()) * len(data)
        kld_loss += kld.item() * len(data)
        reconstruction_loss += recons.item() * len(data)
        optimizer.step()

    if writer:
        writer.add_scalar('Train/Temperature', temperature, writer_epoch)
        writer.add_scalar('Loss/Train/BetaTotal', train_loss / len(train_loader.dataset), writer_epoch)
        writer.add_scalar('Loss/Train/TrueTotal', train_loss_true / len(train_loader.dataset), writer_epoch)
        writer.add_scalar('Loss/Train/Reconstruction', reconstruction_loss / len(train_loader.dataset), writer_epoch)
        writer.add_scalar('Loss/Train/KLD', kld_loss / len(train_loader.dataset), writer_epoch)

    if number_has_leading_zeros_p(epoch):
        print('====> Epoch: {} Average loss: {:.4f}, KLD loss: {}, Reconstruction loss: {}'.format(
            epoch, train_loss / len(train_loader.dataset), kld_loss / len(train_loader.dataset),
            reconstruction_loss / len(train_loader.dataset)))

    return train_loss / len(train_loader.dataset), kld_loss / len(train_loader.dataset), \
           reconstruction_loss / len(train_loader.dataset)


def test_epoch(model, test_loader, epoch, max_temperature=5.0, min_temperature=0.5, hard=False, sigma=0.1, beta=1.0,
               loss_fn="BCE", writer=None, writer_epoch=None, episode=0):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    test_loss_true = 0
    kld_loss = 0
    reconstruction_loss = 0

    with torch.no_grad():
        temperature = min_temperature

        for i, data in enumerate(test_loader):
            data = data.to(device)

            recon_batch, qz, z = model(data, temperature, hard)
            loss, recons, kld, mse = loss_function(recon_batch, data, qz, z, epoch, choose_loss=loss_fn,
                                                   sigma=sigma, beta=beta)
            test_loss += loss.item() * len(data)
            test_loss_true += (recons.item() + kld.item()) * len(data)
            kld_loss += kld.item() * len(data)
            reconstruction_loss += recons.item() * len(data)

            if writer and i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch[:n]])
                grid = torchvision.utils.make_grid(comparison.data.cpu(), nrow=n)
                writer.add_image('Test/', grid, episode)

        if writer:
            writer.add_scalar('Loss/Test/BetaTotal', test_loss / len(test_loader.dataset), writer_epoch)
            writer.add_scalar('Loss/Test/TrueTotal', test_loss_true / len(test_loader.dataset), writer_epoch)
            writer.add_scalar('Loss/Test/Reconstruction', reconstruction_loss / len(test_loader.dataset),
                              writer_epoch)
            writer.add_scalar('Loss/Test/KLD', kld_loss / len(test_loader.dataset), writer_epoch)


        test_loss /= len(test_loader.dataset)
        kld_loss /= len(test_loader.dataset)
        reconstruction_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}, kld loss: {}, reconstruction loss: {}'.format(test_loss, kld_loss,
                                                                                      reconstruction_loss))

    return test_loss, kld_loss, reconstruction_loss


class No_Activation_Res_3d_Conv_15(Res_3d_Conv_15):

    def __init__(self, latent, block, image_size, temp, image_channels=1):
        super().__init__(latent, block, image_size, temp, image_channels)

        dropout = 0.2
        leaky = 0.01
        channels = 64

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.latent, channels, 3, stride=2),
            nn.LeakyReLU(leaky),
            block(channels, dropout=dropout),
            nn.ConvTranspose2d(channels, channels, 4, stride=2),
            nn.LeakyReLU(leaky),
            block(channels, dropout=dropout),
            nn.ConvTranspose2d(channels, self.image_channels, 4, stride=2),  # 86x86
            CropImage((image_size, image_size)),
        )





