from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
import cv2

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode
normal_images = []
potholes_images = []
for dirname, _, filenames in os.walk('Data/normal'):
    for filename in filenames:

        img = cv2.imread(os.path.join(dirname, filename),
                         cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (32, 32))
        print(filename)
        normal_images.append(np.array(img))


train_loader = torch.utils.data.DataLoader(normal_images, batch_size=12,
                                           shuffle=True, num_workers=2)
dataiter = next(iter(train_loader))
images = dataiter.next()
print(images.shape)

plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r')
