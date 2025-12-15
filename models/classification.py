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

# taken from HW5
device = torch.device("cuda")

# ResNet Model
class ResNetModel(nn.Module):
    def __init__(self, num_target_classes, freeze_backbone=False):
        super(ResNetModel, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
        self.model.fc = nn.Linear(512, num_target_classes)
        self.freeze_backbone = freeze_backbone

        if self.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(ResNetModel, self).train(mode)
        if self.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.fc.parameters():
                param.requires_grad = True

    def evaluate(self):
        self.train(False)

    def forward(self, x):
        return self.model(x)
    
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=37, dropout = 0.5):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(p=0.2),

            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(p=0.2),

            nn.Conv2d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(256 * 6 * 6, 512),
            nn.Softplus(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        N, c, H, W = x.shape
        features = self.features(x)
        pooled_features = self.avgpool(features)
        output = self.classifier(torch.flatten(pooled_features, 1))
        return output

def plot_results(epochs, train, val, metric_name="Loss"):
    plt.plot(epochs, train, label="train")
    plt.plot(epochs, val, label="val")
    plt.title(metric_name + " Plot")
    plt.ylabel(metric_name)
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()

def save_checkpoint(save_dir, model, save_name = 'best_model.pth'):
    save_path = os.path.join(save_dir, save_name)
    torch.save(model.state_dict(), save_path)

def load_model(model, save_dir, save_name = 'best_model.pth'):
    save_path = os.path.join(save_dir, save_name)
    model.load_state_dict(torch.load(save_path))
    return model

def train_clas(train_loader, model, criterion, optimizer, epoch, fake_weight_max=0.5):
    model.train()
    losses = []

    it_train = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc="Training ...",
        position=0
    )

    for i, (images, labels, is_fake) in it_train:
        images = images.to(device)
        labels = labels.to(device)
        is_fake = is_fake.to(device)

        optimizer.zero_grad()

        pred = model(images)

        # ---- per-sample loss ----
        loss = criterion(pred, labels)

        # real = 1.0, fake = fake_weight
        fake_weight = fake_weight_max # max(fake_weight_max, 1.0 - epoch / 200)
        weights = torch.where(
            is_fake == 1,
            torch.full_like(is_fake, fake_weight, dtype=torch.float),
            torch.ones_like(is_fake, dtype=torch.float)
        )

        loss = (loss * weights).mean()

        loss.backward()
        optimizer.step()

        losses.append(loss.detach())

        it_train.set_description(f'loss: {loss.item():.3f}')

    return torch.stack(losses).mean().item()


def test_clas(test_loader, model, criterion):
    model.eval()
    losses = []
    correct = 0
    total = 0

    # TO DO: read this documentation and then uncomment the line below; https://pypi.org/project/tqdm/
    it_test = tqdm(enumerate(test_loader), total=len(test_loader), desc="Validating ...", position = 0)
    for i, (images, labels, _) in it_test:

      images, labels = images.to(device), labels.to(device)

      with torch.no_grad():
          pred = model(images)

      loss = criterion(pred, labels)
      losses.append(loss.item())

      label_pred = pred.argmax(-1)
      correct += (label_pred == labels).sum()
      total += len(label_pred)

    mean_accuracy = correct / total
    test_loss = np.mean(losses)
    print('Mean Accuracy: {0:.4f}'.format(mean_accuracy))
    print('Avg loss: {}'.format(test_loss))

    return mean_accuracy, test_loss