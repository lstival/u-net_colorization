# Importing all function of the trainin file
from train import *

# Loss imports
from piq import ssim, SSIMLoss

# ---------------- Read the Data ---------------
# ----------------------------------------------
dataLoader = ld.ReadData()

# Spatial size of training images. All images will be resized to this
image_size = (512,768)

# Batch size during training
batch_size = 5

# Root directory for dataset
dataroot = "../data/train/initial_argument"

dataloader = dataLoader.create_dataLoader(dataroot, image_size)

#Example image
img_example = dataLoader.img_example(dataloader)
img_example
print(f"Sucess to create the dataloader, batch size: {img_example.size}")

# -------------- Instance the model ------------
# ----------------------------------------------

import model as mdl

model = mdl.ViT_UNet(in_ch=1)
# print(type(model))
# y = next(iter(dataloader))[0]
# out = model(y[:,:1,:,:])
# print(out.shape)

# --------------------- Loss -------------------
# ----------------------------------------------

# ssim_index: torch.Tensor = ssim(y, out, data_range=1.)
# loss = SSIMLoss(data_range=1.)
criterion = SSIMLoss(data_range=1.)
# diff = torch.Tensor = loss(y, out)
# diff.backward()
# print(f"SSIM loss: {diff}")

# Metrics of the training
max_lr = 1e-3
epoch = 2
weight_decay = 1e-4

# Doing the training
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epoch,
                                            steps_per_epoch=len(dataloader))

history = fit(epoch, model, dataloader, criterion, optimizer, sched)