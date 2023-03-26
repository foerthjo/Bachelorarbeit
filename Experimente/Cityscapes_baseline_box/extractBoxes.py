import glob
import torch
from torchvision.io import read_image
from torchvision.io import write_png
from Devices import device, cpu
from LabelIDs import nameToLabelID, trainIDsByLabelID
import numpy as np
import json

gtPaths = glob.glob("G:\\cityscapes_dataset\\gtFine_trainvaltest\\gtFine\\train\\*\\*_gtFine_polygons.json")

targetHeight, targetWidth = 256, 512

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

def drawBox(tensor, box, value):
	batchSize, channels, height, width = tensor.size()

	column = torch.arange(0, height, device=device).view([height, 1])
	row = torch.arange(0, width, device=device).view([1, width])

	x, y, endX, endY = box['x'], box['y'], box['endX'], box['endY']
	row_mask = torch.logical_and(row >= y, row < endY).type(torch.float32)
	column_mask = torch.logical_and(column >= x, column < endX).type(torch.float32)
	box_mask = torch.mm(column_mask, row_mask)
	box_mask = box_mask.type(torch.bool).view([1, 1, height, width])
	
	tensor.masked_fill_(box_mask, value)

def loadGT(path):
	file = {}
	with open(path, 'r') as f:
		file = json.load(f)

	imgHeight = file["imgHeight"]
	imgWidth = file["imgWidth"]
	boxes = boxesFromPolygons(file["objects"], imgHeight, imgWidth)
	
	labels = torch.zeros((1, 1, targetHeight, targetWidth), device=device, dtype=torch.long)
	for box in boxes:
		if not box["trainID"] == 0:
			drawBox(labels, box, box["trainID"])

	return labels

imgIndex = 0
for path in gtPaths:
	gt = loadGT(path)
	write_png(gt.view([1, targetHeight, targetWidth]).type(torch.uint8).to(cpu), "G:\\cityscapes_dataset\\train\\boxes\\boxes_" + str(imgIndex) + ".png", )
	print("saved G:\\cityscapes_dataset\\train\\boxes\\boxes_" + str(imgIndex) + ".png")
	imgIndex += 1