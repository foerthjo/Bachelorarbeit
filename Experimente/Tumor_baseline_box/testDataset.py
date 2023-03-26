from Dataset import Dataset
from torchvision.io import write_png
import torch
from Devices import cpu

path_to_gt = "G:\\medical_decathlon_dataset\\Task01_BrainTumour\\boxes"
gt_to_image_path = lambda path: path.replace("boxes\\gt_", "images\\image_")
start = 0
end = 1
batchSize = 8
shuffle = False
repeat = False
dataset = Dataset(path_to_gt, gt_to_image_path, start, end, batchSize, shuffle, repeat)

print(dataset.imageCount)

dataset.loadBatch()

input, segmentation = dataset.loadBatch()
print(input.shape)
print(segmentation.shape)
print(torch.max(input))
print(torch.unique(segmentation))

write_png((input[7][0:3] * 255).type(torch.uint8).to(cpu), "inputXYZ.png", 9)
write_png((segmentation[7] / 3.0 * 255).view([1, 240, 240]).type(torch.uint8).to(cpu), "segmentationMask.png", 9)