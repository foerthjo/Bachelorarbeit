from Dataset import Dataset
from torchvision.io import write_png
import torch
from Devices import cpu

path_to_gt = "G:\\medical_decathlon_dataset\\Task01_BrainTumour\\gts_png"
lambda_gt_to_image_path = lambda path: path.replace("gts_png\\gt_", "images\\image_")
start = 0
end = .1
batchSize = 8
shuffle = False
repeat = False
dataset = Dataset(path_to_gt, start, end, batchSize, shuffle, repeat)

print(dataset.imageCount)

for i in range(6):
	dataset.loadBatch()

input, segmentation, box1, box2, box3 = dataset.loadBatch()
print(input.shape)
print(segmentation.shape)
print(box1.shape)
print(torch.max(input))
print(torch.unique(segmentation))

write_png((input[7][0:3] * 255).type(torch.uint8).to(cpu), "inputXYZ.png", 9)
write_png((segmentation[7] / 3.0 * 255).type(torch.uint8).to(cpu), "segmentationMask.png", 9)
write_png((box2[7] * 255).type(torch.uint8).to(cpu), "boxMask.png", 9)