import glob
import torch
from torchvision.io import read_image
from torchvision.io import write_png
from Devices import device, cpu

from LabelIDs import mapLabelIDsToTrain, ignore_index

height, width = 240, 240

gtPaths = glob.glob("G:\\medical_decathlon_dataset\\Task01_BrainTumour\\gts_reduced\\*.png")

def mask_to_box(mask, label):
	y, x = torch.nonzero(mask == label, as_tuple=True)
	if (y.shape[0] == 0 or x.shape[0] == 0):
		return None
	if (torch.min(x) == 0 and torch.min(y) == 0 and torch.max(x) == 0 and torch.max(y) == 0):
		return None
	return {
		"x": int(torch.min(x).to(cpu)),
		"y": int(torch.min(y).to(cpu)),
		"endX": int(torch.max(x).to(cpu)),
		"endY": int(torch.max(y).to(cpu)),
	}

def scaleBoxDown(box):
	width = box["endX"] - box["x"]
	height = box["endY"] - box["y"]
	border = min(width * .2, height * .2)
	box["x"] += border
	box["y"] += border
	box["endX"] -= border
	box["endY"] -= border

def drawBox(tensor, box, value):
	if box == None:
		return
	
	channels, height, width = tensor.size()

	column = torch.arange(0, height, device=device).view([height, 1])
	row = torch.arange(0, width, device=device).view([1, width])

	x, y, endX, endY = box['x'], box['y'], box['endX'], box['endY']
	row_mask = torch.logical_and(row >= x, row < endX).type(torch.float32)
	column_mask = torch.logical_and(column >= y, column < endY).type(torch.float32)
	box_mask = torch.mm(column_mask, row_mask)
	box_mask = box_mask.type(torch.bool).view([1, height, width])
	
	tensor.masked_fill_(box_mask, value)

for path in gtPaths:
	name = path.split("\\")[-1]
	gt = mapLabelIDsToTrain(read_image(path).to(device).view((height, width)))
	boxlabels = torch.zeros((1, height, width), device=device, dtype=torch.long)

	box1 = mask_to_box(gt, 1)
	box2 = mask_to_box(gt, 2)
	if not box1 == None:
		drawBox(boxlabels, box1, ignore_index)
		scaleBoxDown(box1)
		drawBox(boxlabels, box1, 1)
	
	if not box2 == None:
		drawBox(boxlabels, box2, ignore_index)
		scaleBoxDown(box2)
		drawBox(boxlabels, box2, 2)
	
	print(name)
	write_png((boxlabels).type(torch.uint8).to(cpu), "G:\\medical_decathlon_dataset\\Task01_BrainTumour\\boxes_i\\" + name, compression_level=9)