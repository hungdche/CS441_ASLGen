import os
import random
from natsort import natsorted
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from .diffaugment import DiffAugment

device = torch.device("cuda")

# ----------------- MODELS ----------------- #

class TinyVAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.is_conditional = cfg['is_conditional']
        self.is_vq = cfg.get('is_vq', False)
        
        latent_dim = cfg['latent_dim']
        input_res = cfg['input_res']

        # encoder
        enc_layers = []
        in_ch = cfg['img_channels']
        if self.is_conditional:
            in_ch += cfg['num_classes']

        for out_ch in cfg['enc_sizes']:
            enc_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                    nn.GroupNorm(8, out_ch),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            in_ch = out_ch
        self.encoder = nn.Sequential(*enc_layers)
        self.final_h = input_res // (2 ** len(cfg['enc_sizes']))
        self.final_w = input_res // (2 ** len(cfg['enc_sizes']))
        self.final_ch = cfg['enc_sizes'][-1]

        # reparam trick
        self.fc_mu = nn.Linear(self.final_ch * self.final_h * self.final_w, latent_dim)
        self.fc_logvar = nn.Linear(self.final_ch * self.final_h * self.final_w, latent_dim)

        # decoder
        self.dec_init_ch = self.final_ch
        if self.is_conditional:
            latent_dim += cfg['num_classes']
        self.fc_dec = nn.Linear(latent_dim, self.dec_init_ch * self.final_h * self.final_w)
        dec_layers = []

        # decoder blocks
        in_ch = self.dec_init_ch
        for out_ch in cfg['dec_sizes']:
            dec_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                    nn.GroupNorm(8, out_ch),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            in_ch = out_ch

        # decode to image
        dec_layers.append(
            nn.ConvTranspose2d(
                in_channels=in_ch,
                out_channels=cfg['img_channels'],
                kernel_size=4,
                stride=2,
                padding=1
            )
        )

        self.decoder = nn.Sequential(*dec_layers)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x, c=None):
        if self.is_conditional:
            assert c is not None, "Conditional VAE requires class labels"
            c_onehot = F.one_hot(c, num_classes=self.cfg['num_classes']).float().to(x.device)
            c_onehot = c_onehot.unsqueeze(2).unsqueeze(3)
            c_onehot = c_onehot.expand(-1, -1, x.size(2), x.size(3))
            x = torch.cat([x, c_onehot], dim=1)

        x = self.encoder(x)
        x = x.flatten(start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z, c=None, c_onehot=None):
        if self.is_conditional:
            assert c is not None or c_onehot is not None, "Conditional VAE requires class labels"
            if c_onehot is None:
                c_onehot = F.one_hot(c, num_classes=self.cfg['num_classes']).float().to(z.device)
            z = torch.cat([z, c_onehot], dim=1)
            
        x = self.fc_dec(z)
        x = x.view(-1, self.final_ch, self.final_h, self.final_w)
        x = self.decoder(x)
        return torch.sigmoid(x)

    def forward(self, x, c=None):
        if self.is_conditional:
            assert c is not None, "Conditional VAE requires class labels"
        mu, logvar = self.encode(x, c=c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c=c)
        return recon, mu, logvar
    
    def sample(self, num_samples, cs=None):
        if self.is_conditional:
            assert cs is not None, "Conditional VAE requires class labels"
            assert len(cs) == num_samples, "Number of class labels must match number of samples"

        z = torch.randn(num_samples, self.cfg['latent_dim']).to(next(self.parameters()).device)
        samples = self.decode(z, c=cs)
        return samples
    
    def load_from_ckpt(self, ckpt_path):
        self.load_state_dict(torch.load(ckpt_path))
    
lpips_fn = LearnedPerceptualImagePatchSimilarity(
    normalize=True, net_type="vgg", sync_on_compute=False
).to(device)

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = F.l1_loss(recon_x, x, reduction='mean')
    # lpips_loss = lpips_fn(recon_x, x).mean()
    # recon_loss = recon_loss + 0.0 * lpips_loss
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


# _----------------- TRAINERS ----------------- #\

def train_vae(train_loader, model, criterion, optimizer, beta, alphabet_class, epoch, use_diffaugment=False):
    model.train()
    loss_ = 0.0
    losses = {
        'recon_loss': [],
        'kl_loss': []
    }

    it_train = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training ...", position = 0) # progress bar
    for i, (images, labels) in it_train:

        # move to device
        images = images.to(device)
        labels = labels.to(device)

        # zero gradients
        optimizer.zero_grad()

        # forward pass
        if not use_diffaugment:
            recon, mu, logvar = model(images, labels)

            # compute loss + kl divergence
            loss, recon_loss, kl_loss = criterion(recon, images, mu, logvar, beta=beta)

            # backward pass
            loss.backward()

            # update weights
            optimizer.step()

            # track loss
            losses['recon_loss'].append(recon_loss.item())
            losses['kl_loss'].append(kl_loss.item())

            it_train.set_description(f"epoch: {epoch}; l_recon: {losses['recon_loss'][-1]:.3f}; l_kl: {losses['kl_loss'][-1]:.3f}")

        else:
            
            # apply diffaugment
            x_aug_input = DiffAugment(images, policy='color,translation')
            x_aug_target = DiffAugment(images, policy='color,translation')

            recon, mu, logvar = model(x_aug_input, labels)

            # compute loss + kl divergence
            loss, recon_loss, kl_loss = criterion(recon, x_aug_target, mu, logvar, beta=beta)

            # backward pass
            loss.backward()

            # update weights
            optimizer.step()

            # track loss
            losses['recon_loss'].append(recon_loss.item())
            losses['kl_loss'].append(kl_loss.item())

            it_train.set_description(f"epoch: {epoch}; l_recon: {losses['recon_loss'][-1]:.3f}; l_kl: {losses['kl_loss'][-1]:.3f}")

    # log recon-gt images pair 
    if epoch % 100 == 0:
        logdir = f'training_res/vae/{alphabet_class}/training/epoch_{epoch}'
        os.makedirs(logdir, exist_ok=True)
        for i in range(images.size(0)):
            # concat them together
            final_img = torch.cat([
                recon[i].cpu(), images[i].cpu()
            ], dim=-1)
            if use_diffaugment:
                final_img = torch.cat([
                    final_img, x_aug_input[i].cpu(), x_aug_target[i].cpu()
                ], dim=-1)
            output_img = transforms.ToPILImage()(final_img)
            output_img.save(os.path.join(logdir, f'{i}.png'))

    return {k: np.array(v).mean() for k, v in losses.items()}

def test_vae(model, save_img_folder, num_samples=16, compute_fid=False, img_folder=None):
    assert not (compute_fid and img_folder is None), "To compute FID, img_folder must be provided"
    model.eval()

    # sample a bunch new samples  
    with torch.no_grad():
        # randomly sample labels 
        cs = torch.randint(0, model.cfg['num_classes'], (num_samples,))
        samples = model.sample(num_samples, cs).cpu()

    # save samples 
    os.makedirs(save_img_folder, exist_ok=True)
    for i in range(num_samples):
        sample_img = transforms.ToPILImage()(samples[i])
        sample_img.save(os.path.join(save_img_folder, f'sample_{i}.png'))

    if compute_fid:
        from cleanfid import fid
        score = fid.compute_fid(save_img_folder, img_folder)
        return score

    # save model
    torch.save(model.state_dict(), os.path.join(save_img_folder, 'model.pth'))

    return None