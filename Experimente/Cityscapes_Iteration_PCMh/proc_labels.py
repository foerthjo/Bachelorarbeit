import torch
from torchvision.io import read_image, write_png
import glob
from Devices import device, cpu
from torchvision.models.segmentation import deeplabv3_resnet50
import json
import numpy as np

from LabelIDs import num_classes, nameToLabelID, trainIDsByLabelID, mapLabelIDsToTrain, ignore_index

height, width = 256, 512
targetHeight, targetWidth = height, width

model = deeplabv3_resnet50(weights=None, num_classes=num_classes).to(device)
model.load_state_dict(torch.load("teacher.ckpt"))
model.eval().to(device)

gtPaths = glob.glob("G:\\cityscapes_dataset\\gtFine_trainvaltest\\gtFine\\train\\*\\*_gtFine_polygons.json")

def boxesFromPolygons(objects, imgHeight, imgWidth):
	boxes = []
	for obj in objects:
		labelName = obj["label"]
		labelID = nameToLabelID(labelName)
		trainID = trainIDsByLabelID[labelID]
		vertices = np.array(obj["polygon"]) * targetHeight / imgHeight
		min = np.amin(vertices, axis=0)
		max = np.amax(vertices, axis=0)
		x = min[1]
		endX = max[1]
		width = endX - x
		y = min[0]
		endY = max[0]
		height = endY - y
		area = width * height
		boxes.append({
			"labelName": labelName,
			"labelID": labelID,
			"trainID": trainID,
			"y": y,
			"x": x,
			"endY": endY,
			"endX": endX,
			"height": height,
			"width": width,
			"area": area,
		})
	
	boxes.sort(key=lambda box: -box["area"])
	return boxes

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
	column = torch.arange(0, height, device=device).view([height, 1])
	row = torch.arange(0, width, device=device).view([1, width])

	x, y, endX, endY = box['x'], box['y'], box['endX'], box['endY']
	row_mask = torch.logical_and(row >= y, row < endY).type(torch.float32)
	column_mask = torch.logical_and(column >= x, column < endX).type(torch.float32)
	box_mask = torch.mm(column_mask, row_mask)
	box_mask = box_mask.type(torch.bool).view([1, height, width])
	
	tensor.masked_fill_(box_mask, value)

def doBoxesOverlap(box1, box2):
	return box1['x'] <= box2['endX'] and box2['x'] <= box1['endX'] and box1['y'] <= box2['endY'] and box2['y'] <= box1['endY']

index = 0
for path in gtPaths:
	name = "labels_" + str(index) + ".png"

	input = read_image("G:\\cityscapes_dataset\\train\\images_png\\image_" + str(index) + ".png").to(device).view([1, 3, height, width]) / 255.0
	
	result = torch.argmax(model(input)['out'], dim=1, keepdim=False).view([1, height, width])
	# write_png((result).type(torch.uint8).to(cpu), "output_" + name, compression_level=9)

	file = {}
	with open(path, 'r') as f:
		file = json.load(f)

	imgHeight = file["imgHeight"]
	imgWidth = file["imgWidth"]
	boxes = boxesFromPolygons(file["objects"], imgHeight, imgWidth)


	boxgt_inner = torch.zeros((1, height, width), device=device, dtype=torch.long)

	for box in boxes:
		if not box["trainID"] == 0:
			drawBox(boxgt_inner, box, ignore_index)
	
	for box in boxes:
		if not box["trainID"] == 0:
			drawBox(boxgt_inner, scaleBoxDown(box), box["trainID"])

	wrong_mask = torch.zeros((1, height, width), device=device, dtype=torch.uint8)
	for i in range(num_classes):
		boxgt_class = torch.zeros((1, height, width), device=device, dtype=torch.long)
		for box in boxes:
			if box["trainID"] == i:
				drawBox(boxgt_class, box, 1)
		
		wrong_mask += (1 - boxgt_class) * (result == i).type(torch.uint8)
	
	wrong_mask = wrong_mask > 0
	result[wrong_mask] = boxgt_inner[wrong_mask]
	

	for i in range(0, len(boxes)):
		box = boxes[i]
		boxgt_area = area(box)
		if boxgt_area > 0:
			boxgt_full = torch.zeros((1, height, width), device=device, dtype=torch.uint8)
			drawBox(boxgt_full, box, 1)
			overlap = (boxgt_full * (result == box['trainID'])).sum().item() / boxgt_area
			if (overlap < .2):
				reset_mask = torch.zeros([1, height, width], device=device, dtype=torch.uint8)
				downScaled = scaleBoxDown(box)
				drawBox(reset_mask, downScaled, 1)
				for e in range(i + 1, len(boxes)):
					other = boxes[e]
					if doBoxesOverlap(downScaled, other):
						drawBox(reset_mask, other, 0)
				
				result.masked_fill_(reset_mask, box['trainID'])


	print(name)
	write_png((result).type(torch.uint8).to(cpu), "G:\\cityscapes_dataset\\train\\iteration_postpro_1\\" + name, compression_level=9)
	index += 1