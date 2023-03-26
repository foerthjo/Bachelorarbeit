import glob
import random
from torchvision.io import read_image
from Devices import device
from torchvision import transforms
import torch
import math
from LabelIDs import mapLabelIDsToTrain

width, height = 240, 240

def loadImg(pathObj):
	imageXYZ = read_image(pathObj["imageXYZPath"]).to(device) / 255.0
	imageW = read_image(pathObj["imageWPath"]).to(device) / 255.0
	c, height, width = imageXYZ.shape
	return torch.cat([imageXYZ, imageW]).view([1, 4, width, height])

def loadGT(pathObj):
	gt = read_image(pathObj["gtPath"]).to(device).type(torch.long).view([1, width, height])
	if pathObj["mapIDs"]:
		gt = mapLabelIDsToTrain(gt)
	return gt

class Dataset:
	def __init__(this, path_to_gt, gt_to_image_path, start, end, batchSize, shuffle, repeat, map_IDs):
		gtPaths = glob.glob(path_to_gt + '\\*.png')
		gtPaths = list(map(lambda path: path.replace('/', '\\'), gtPaths))
		gtPaths.sort(key=lambda path: int(path.split("\\")[-1].replace("gt_", "").replace(".png", "")), reverse=False)
		gtPaths = gtPaths[int(len(gtPaths) * start):int(len(gtPaths) * end)]

		this.imageCount = len(gtPaths)
		this.batchCount = math.ceil(this.imageCount / batchSize)

		this.pathObjs = list(map(lambda gtPath: {
			"gtPath": gtPath,
			"imageXYZPath": gt_to_image_path(gtPath).replace("images\\image_", "imagesXYZ_png\\imageXYZ_"),
			"imageWPath": gt_to_image_path(gtPath).replace("images\\image_", "imagesW_png\\imageW_"),
			"mapIDs": map_IDs,
		}, gtPaths))

		this.index = 0
		this.batchSize = batchSize

		this.shuffle = shuffle
		if shuffle:
			print('shuffling dataset')
			random.shuffle(this.pathObjs)
		
		this.repeat = repeat
	
	def printPaths(this):
		print(len(this.pathObjs))
		print(this.pathObjs[0:3])

	def loadBatch(this):
		i = 0
		pathObjs = []
		while i < this.batchSize:
			if this.index >= this.imageCount:
				if (not this.repeat):
					break
				
				this.reset()

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