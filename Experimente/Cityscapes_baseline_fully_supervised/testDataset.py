from Dataset import Dataset
from torchvision.io import write_png
import torch
from Devices import cpu
from LabelIDs import num_classes

path_to_gt = "G:\\cityscapes_dataset\\train\\gts_png"
start = 0
end = 1
batchSize = 8
shuffle = False
dataset = Dataset(path_to_gt, start, end, batchSize, shuffle)

print(dataset.imageCount)

for i in range(6):
	dataset.loadBatch()

input, segmentation = dataset.loadBatch()
b, c, h, w = input.shape
print(input.shape)
print(segmentation.shape)
print(torch.max(input))
print(torch.unique(segmentation))

write_png((input[7] * 255).type(torch.uint8).to(cpu), "input.png", 9)
write_png((segmentation[7].type(torch.float) / num_classes * 255).view(1, h, w).type(torch.uint8).to(cpu), "segmentationMask.png", 9)