import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
# this is for the commit


def train_model(model, train_loader):
    if torch.cuda.is_available():
        train_on_gpu = "True"
        print("cuda")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # number of epochs to train the model
    n_epochs = 30

    for epoch in range(1, n_epochs+1):

        # keep track of training and validation loss
        train_loss = 0.0

        ###################
        # train the model #
        ###################
        model.train()
        for data, target in train_loader:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item()*data.size(0)

        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch, train_loss))

        # save model if validation loss has decreased
