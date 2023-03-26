import torch
from torchvision.io import read_image, write_png
import glob
from Devices import device, cpu
from Unet import Unet

from LabelIDs import num_classes, mapLabelIDsToTrain, ignore_index

height, width = 240, 240

model = Unet(num_classes)
model.load_state_dict(torch.load("teacher.ckpt"))
model.eval().to(device)

gtPaths = glob.glob("G:\\medical_decathlon_dataset\\Task01_BrainTumour\\gts_reduced\\*.png")
gtPaths = list(map(lambda path: path.replace('/', '\\'), gtPaths))
gtPaths.sort(key=lambda path: int(path.split("\\")[-1].replace("gt_", "").replace(".png", "")), reverse=False)
gtPaths = gtPaths[0:int(len(gtPaths) * .3)]

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

def area(box):
	width = box["endX"] - box["x"]
	height = box["endY"] - box["y"]
	return width * height
	
def scaleBoxDown(box):
	width = box["endX"] - box["x"]
	height = box["endY"] - box["y"]
	border = min(width * .2, height * .2)
	return {
		"x": box["x"] + border,
		"y": box["y"] + border,
		"endX": box["endX"] - border,
		"endY": box["endY"] - border,
	}

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

	imageXYZ = read_image(path.replace("gts_reduced\\gt_", "imagesXYZ_png\\imageXYZ_")).to(device) / 255.0
	imageW = read_image(path.replace("gts_reduced\\gt_", "imagesW_png\\imageW_")).to(device) / 255.0
	input = torch.cat([imageXYZ, imageW]).view([1, 4, height, width])
	
	
	result = torch.argmax(model(input), dim=1, keepdim=False).view([1, height, width])
	# write_png((result).type(torch.uint8).to(cpu), "output_" + name, compression_level=9)


	gt = mapLabelIDsToTrain(read_image(path).to(device).view((height, width)))
	box1 = mask_to_box(gt, 1)
	box2 = mask_to_box(gt, 2)

	boxgt_inner = torch.zeros((1, height, width), device=device, dtype=torch.long)
	boxgt_1 = torch.zeros((1, height, width), device=device, dtype=torch.long)
	boxgt_2 = torch.zeros((1, height, width), device=device, dtype=torch.long)

	if not box1 == None:
		drawBox(boxgt_inner, box1, ignore_index)
		drawBox(boxgt_inner, scaleBoxDown(box1), 1)
		drawBox(boxgt_1, box1, 1)
	
	if not box2 == None:
		drawBox(boxgt_inner, box2, ignore_index)
		drawBox(boxgt_inner, scaleBoxDown(box2), 2)
		drawBox(boxgt_2, box2, 1)

	wrong_mask = torch.zeros((1, height, width), device=device, dtype=torch.uint8)
	wrong_mask += (1 - boxgt_1) * (result == 1).type(torch.uint8)
	wrong_mask += (1 - boxgt_2) * (result == 2).type(torch.uint8)
	wrong_mask = wrong_mask > 0
	result[wrong_mask] = boxgt_inner[wrong_mask]
	

	if not box1 == None:
		boxgt_area_1 = area(box1)
		if boxgt_area_1 > 0:
			inf_area_1 = (result == 1).type(torch.uint8).sum().item()
			overlap = inf_area_1 / boxgt_area_1
			if overlap < .25:
				mask_1 = torch.zeros([1, height, width], device=device, dtype=torch.uint8)
				drawBox(mask_1, scaleBoxDown(box1), 1)
				drawBox(mask_1, box2, 0)
				result.masked_fill_(mask_1.type(torch.bool), 1)
	
	if not box2 == None:
		boxgt_area_2 = area(box2)
		if boxgt_area_2 > 0:
			inf_area_2 = (result == 2).type(torch.uint8).sum().item()
			overlap = inf_area_2 / boxgt_area_2
			if overlap < .25:
				drawBox(result, scaleBoxDown(box2), 2)


	print(name)
	write_png((result).type(torch.uint8).to(cpu), "G:\\medical_decathlon_dataset\\Task01_BrainTumour\\iteration_postpro_2_1\\" + name, compression_level=9)