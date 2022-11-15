
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
print("sucess")