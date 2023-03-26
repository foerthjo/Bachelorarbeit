import glob
import torchvision
from torchvision.io import write_png
from torchvision.io import read_image
from LabelIDs import mapLabelIDsToTrain
import torch

gtPaths = glob.glob("G:\\cityscapes_dataset\\gtFine_trainvaltest\\gtFine\\val\\*\\*_gtFine_labelIds.png")

targetHeight, targetWidth = 256, 512

resizeTransform = torchvision.transforms.Resize([targetHeight, targetWidth], interpolation=torchvision.transforms.InterpolationMode.NEAREST)

imgIndex = 0
for path in gtPaths:
	gt = resizeTransform(read_image(path).to('cuda'))
	gt = mapLabelIDsToTrain(gt)
	write_png(gt.type(torch.uint8).to('cpu'), "G:\\cityscapes_dataset\\val\\trainIDs_png\\gt_" + str(imgIndex) + ".png", 9)
	print("saved G:\\cityscapes_dataset\\train\\trainIDs\\gt_" + str(imgIndex) + ".png")
	imgIndex += 1