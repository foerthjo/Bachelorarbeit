import glob
import random
from torchvision.io import read_image
from Devices import device
import torch
from LabelIDs import mapLabelIDsToTrain
import math

width, height = 512, 256

def loadImg(pathObj):
	image = read_image(pathObj["imagePath"]).to(device).type(torch.float) / 255.0
	return image.view((1, 3, height, width))

def loadGT(pathObj):
	gt = read_image(pathObj["gtPath"]).to(device).type(torch.long)
	gt = gt.view((1, height, width))
	return mapLabelIDsToTrain(gt)

class Dataset:
	def __init__(this, path_to_gt, start, end, batchSize, shuffle):
		gtPaths = glob.glob(path_to_gt + '\\*.png')
		gtPaths = list(map(lambda path: path.replace('/', '\\'), gtPaths))
		gtPaths.sort(key=lambda path: int(path.split("\\")[-1].replace("gt_", "").replace(".png", "")), reverse=False)
		gtPaths = gtPaths[int(len(gtPaths) * start):int(len(gtPaths) * end)]

		this.imageCount = len(gtPaths)

		this.pathObjs = list(map(lambda gtPath: {
			"gtPath": gtPath,
			"imagePath": gtPath.replace("gts_png\\gt_", "images_png\\image_"),
		}, gtPaths))

		this.index = 0
		this.batchSize = batchSize
		this.batchCount = math.ceil(this.imageCount / batchSize)

		this.shuffle = shuffle
		if shuffle:
			print('shuffling dataset')
			random.shuffle(this.pathObjs)
	
	def printPaths(this):
		print(len(this.pathObjs))
		print(this.pathObjs[0:3])

	def loadBatch(this):
		i = 0
		pathObjs = []
		while i < this.batchSize:
			if this.index >= this.imageCount:
				break

			pathObjs.append(this.pathObjs[this.index])
			this.index += 1
			
			i += 1

		if len(pathObjs) == 0:
			return None
		
		batch = (
			torch.cat(list(map(loadImg, pathObjs))),
			torch.cat(list(map(loadGT, pathObjs))),
		)
		return batch
	
	def resetIndex(this):
		this.index = 0
	
	def reset(this):
		this.index = 0
		if this.shuffle:
			print('shuffling dataset')
			random.shuffle(this.pathObjs)