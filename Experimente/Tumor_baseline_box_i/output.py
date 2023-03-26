import torch
from torch import nn
from torchvision.io import read_image
from torchvision.io import write_png
import torchvision.transforms as transforms
import numpy as np
from Dataset import Dataset
from Devices import device, cpu
import torchvision
from torchvision.models.segmentation import deeplabv3_resnet50
from torchmetrics.classification import MulticlassJaccardIndex
import sys
from Unet import Unet

batchSize = 32

num_classes = 2

model = Unet(num_classes)

path_to_gt = "G:\\medical_decathlon_dataset\\Task01_BrainTumour\\gts_png"
start = 0
end = .1
shuffle = False
repeat = False
dataset = Dataset(path_to_gt, start, end, batchSize, shuffle, repeat)

modelname = 'model'
modelPath = modelname + '.ckpt'
print('loading ' + modelPath)
try:
	model.load_state_dict(torch.load(modelPath))
except:
	print('loading model failed')
	sys.exit()

model = model.to(device)

dataset.loadBatch()
data = dataset.loadBatch()
input, segmentation, box1, box2, box3 = data
localBatchSize, channels, height, width = input.shape

output = model(input).detach()
output = torch.eq(torch.max(output, dim=1, keepdim=True).values, output)

for i in range(localBatchSize):
	image = (input[i][0:3] * 255).type(torch.uint8)
	image = torchvision.utils.draw_segmentation_masks(image.to(cpu), output[i].to(cpu), alpha=0.7)
	write_png(image.to(cpu), 'inference_' + str(i) + '.png', 9)