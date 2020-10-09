import os

import cv2
import numpy as np
from tqdm import tqdm
from random import shuffle
import torch
import matplotlib.pyplot as plt
from skimage import io, transform
# this is for the commit


def data__loader():

    normal_images = []
    potholes_images = []
    for dirname, _, filenames in os.walk('Data/normal'):
        for filename in filenames:

            img = cv2.imread(os.path.join(dirname, filename),
                             cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (32, 32))

            normal_images.append(np.array(img))

    for dirname, _, filenames in os.walk('Data/potholes'):
        for filename in filenames:

            img = cv2.imread(os.path.join(dirname, filename),
                             cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (32, 32))

            potholes_images.append(np.array(img))

    print(len(normal_images))
    print(len(potholes_images))

    processed_data = []
    # t = []
    for img in normal_images:
        # t = torch.LongTensor(1)
        t = np.array([0])
        # img = torch.FloatTensor(img)
        img = np.ndarray(img)
        processed_data.append([img/255, t])
    # t = []
    for img in potholes_images:
        # t = torch.LongTensor(1)
        t = np.array([1])
        # img = torch.FloatTensor(img)
        img = np.ndarray(img)
        processed_data.append([img/255, t])

    print(len(processed_data))
    shuffle(processed_data)

    train_data = processed_data[70:]
    test_data = processed_data[0:70]
    print(f"size of training data {len(train_data)}")
    print(f"size of testing data {len(test_data)}")

    return train_data, test_data


def plot(loader):
    dataiter = iter(loader)
    image, labels = dataiter.next()
    images = image.reshape(64, 1, 50, 50)
    print(images.shape)
    print(labels.shape)

    plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r')
    return images, labels
