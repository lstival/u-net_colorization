
# Load the DataSet
import load_data as ld

dataLoader = ld.ReadData()

# Spatial size of training images. All images will be resized to this
image_size = 256

# Batch size during training
batch_size = 5

# Root directory for dataset
dataroot = "../data/train/initial_argument"

dataloader = dataLoader.create_dataLoader(dataroot, image_size)

#Example image
img_example = dataLoader.img_example(dataloader)
img_example
print(f"sucess dataloader: {img_example.size}")


# instance the model
import encoder as enc
import model as mdl

# model = enc.Enconder()
model = mdl.ViT_UNet()
print(type(model))
print(model(next(iter(dataloader))[0]).shape)
