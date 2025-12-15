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

from .diffaugment import DiffAugment

POLICY = 'color,translation,cutout'
device = torch.device("cuda")

# ----------------- MODELS ----------------- #
class Generator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.init_size = cfg['input_res'] // (2 ** len(cfg['gen_sizes']))
        self.is_conditional = cfg['is_conditional']
        
        if self.is_conditional:
            self.class_emb = nn.Embedding(cfg['num_classes'], cfg['num_classes'])

        input_ch = cfg['latent_dim']
        if self.is_conditional:
            input_ch += cfg['num_classes']

        self.fc = nn.Linear(input_ch, cfg['gen_sizes'][0] * self.init_size * self.init_size)

        self.net = []
        for i in range(len(cfg['gen_sizes'])-1):
            self.net.append(nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(cfg['gen_sizes'][i], cfg['gen_sizes'][i+1], 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ))
        self.net.append(nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(cfg['gen_sizes'][-1], cfg['img_channels'], 3, padding=1),
            nn.Sigmoid()
        ))
        self.net = nn.Sequential(*self.net)


    def forward(self, z, c=None):
        if self.is_conditional:
            assert c is not None, "Conditional GAN requires class labels"
            c_emb = self.class_emb(c)
            z = torch.cat([z, c_emb], dim=1)

        out = self.fc(z)
        out = out.view(z.size(0), self.cfg['gen_sizes'][0], self.init_size, self.init_size)
        return self.net(out)
    
    def load_from_ckpt(self, ckpt_path):
        self.load_state_dict(torch.load(ckpt_path))

class Discriminator(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.is_conditional = cfg['is_conditional']

        input_ch = cfg['img_channels']
        if self.is_conditional:
            input_ch += cfg['num_classes']
            self.class_emb = nn.Embedding(cfg['num_classes'], cfg['num_classes'])

        self.init_size = cfg['input_res'] // (2 ** len(cfg['disc_sizes']))

        self.net = [nn.Sequential(
            nn.Conv2d(input_ch, cfg['disc_sizes'][0], 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )]
        for i in range(len(cfg['disc_sizes'])-1):
            self.net.append(nn.Sequential(
                nn.Conv2d(cfg['disc_sizes'][i], cfg['disc_sizes'][i+1], 4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ))

        self.net = nn.Sequential(*self.net)

        self.fc = nn.Linear(cfg['disc_sizes'][-1] * self.init_size * self.init_size, 1)

    def forward(self, x, c=None):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        if self.is_conditional:
            assert c is not None, "Conditional GAN requires class labels"
            c_emb = self.class_emb(c)
            x = torch.cat([x, c_emb], dim=1)
        return self.fc(x)  # raw logits
    

# _----------------- TRAINERS ----------------- #

gan_criterion = nn.BCEWithLogitsLoss()

def train_gan(
    train_loader,
    G,
    D,
    g_optimizer,
    d_optimizer,
    epoch,
    latent_dim,
    use_diffaugment=True
):
    G.train()
    D.train()

    losses = {
        'd_loss': [],
        'g_loss': []
    }

    it_train = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc="Training GAN ...",
        position=0
    )

    for i, (real_imgs, _) in it_train:
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)

        # Train Discriminator
        z = torch.randn(batch_size, latent_dim, device=device)
        fake_imgs = G(z).detach()

        if use_diffaugment:
            real_in = DiffAugment(real_imgs, policy=POLICY)
            fake_in = DiffAugment(fake_imgs, policy=POLICY)
        else:
            real_in = real_imgs
            fake_in = fake_imgs

        real_logits = D(real_in)
        fake_logits = D(fake_in)

        real_labels = torch.ones_like(real_logits)
        fake_labels = torch.zeros_like(fake_logits)

        d_loss_real = gan_criterion(real_logits, real_labels)
        d_loss_fake = gan_criterion(fake_logits, fake_labels)
        d_loss = d_loss_real + d_loss_fake

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        z = torch.randn(batch_size, latent_dim, device=device)
        fake_imgs = G(z)

        if use_diffaugment:
            fake_in = DiffAugment(fake_imgs, policy=POLICY)
        else:
            fake_in = fake_imgs

        fake_logits = D(fake_in)
        real_labels = torch.ones_like(fake_logits)  # want generator to fool discriminator
        g_loss = gan_criterion(fake_logits, real_labels)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # Logging
        losses['d_loss'].append(d_loss.item())
        losses['g_loss'].append(g_loss.item())

        it_train.set_description(
            f"epoch: {epoch}; "
            f"d_loss: {d_loss.item():.3f}; "
            f"g_loss: {g_loss.item():.3f}"
        )

    # Save samples
    if epoch % 100 == 0:
        save_dir = f"training_res/gan/training/epoch_{epoch}"
        os.makedirs(save_dir, exist_ok=True)

        with torch.no_grad():
            z = torch.randn(16, latent_dim, device=device)
            samples = G(z).cpu()

        for i in range(samples.size(0)):
            img = transforms.ToPILImage()(samples[i])
            img.save(os.path.join(save_dir, f"{i}.png"))

    return {k: np.mean(v) for k, v in losses.items()}


def test_gan(
    G,
    save_img_folder,
    latent_dim,
    num_samples=16,
    compute_fid=False,
    img_folder=None
):
    assert not (compute_fid and img_folder is None), \
        "To compute FID, img_folder must be provided"

    G.eval()
    os.makedirs(save_img_folder, exist_ok=True)

    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim, device=device)
        samples = G(z).cpu()

    for i in range(num_samples):
        img = transforms.ToPILImage()(samples[i])
        img.save(os.path.join(save_img_folder, f"sample_{i}.png"))

    if compute_fid:
        from cleanfid import fid
        return fid.compute_fid(save_img_folder, img_folder)

    torch.save(G.state_dict(), os.path.join(save_img_folder, 'G.pth'))
    return None