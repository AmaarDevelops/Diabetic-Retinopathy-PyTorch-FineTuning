import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.transforms as transforms
import torch
from torchvision.datasets import ImageFolder


# ------------------- Data Transformation and preprocessing ---------------------

transform = transforms.Compose([
    transforms.Resize((256,256)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(degrees=(-10,10)),

    transforms.ColorJitter(0.1,0.2,0.1,0.05),

    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

# Datasets

train_dataset = ImageFolder('./data/processed/Training',transform=transform)

test_dataset = ImageFolder('./data/processed/Testing',transform=transform)

val_dataset = ImageFolder('./data/Validation',transform=transform)

# Loaders

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=32,shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=32,shuffle=False)

val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=32,shuffle=False)



