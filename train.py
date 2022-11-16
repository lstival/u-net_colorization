
# Load the DataSet
import load_data as ld
import torch
# _______________________________
####### Train

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


import time
import os
from tqdm import tqdm

def fit(epochs, model, train_loader, criterion, optimizer, scheduler, patch=False, device="cpu"):

    lrs = []
    train_losses = []
    model.to(device)
    fit_time = time.time()

    for e in range(epochs):
        since = time.time()
        running_loss = 0

        # training loop
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            # training phase
            img, _ = data
            img_gray = img[:,:1,:,:]

            # forward
            output = model(img_gray)
            loss = criterion(output, img)

            # backward
            loss.backward()
            optimizer.step()  # update weight
            optimizer.zero_grad()  # reset gradient

            # step the learning rate
            lrs.append(get_lr(optimizer))
            scheduler.step()

            running_loss += loss.item()

            
            # iou

            print("Epoch:{}/{}..".format(e + 1, epochs),
                  "Train Loss: {:.3f}..".format(running_loss / len(train_loader)),
                  "Time: {:.2f}m".format((time.time() - since) / 60))

    train_losses.append(running_loss / len(train_loader))
    history = {'train_loss': train_losses,
               'lrs': lrs}

    print('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))
    return history
