from Dataset import Dataset
from torchvision.io import write_jpeg
import torch
from Devices import device, cpu
import torchvision

dataset = Dataset(
	'G:/cityscapes_dataset/leftImg8bit_trainvaltest/leftImg8bit/train',
	lambda path: path.replace('leftImg8bit_trainvaltest\\leftImg8bit', 'gtFine_trainvaltest\\gtFine'),
	1, False, False
)

(input, target) = dataset.loadBatch()

write_jpeg((target + 1).type(torch.uint8).to(cpu), 'labelIDs.JPG', 100)

image = (input[0] * 255).type(torch.uint8)
write_jpeg(image.to(cpu), 'image.JPG', 100)