import torch
from Dataset import Dataset
from torchvision.models.segmentation import deeplabv3_resnet50
from torchmetrics.classification import MulticlassJaccardIndex
from Devices import device
import torchvision
from Files import writeFile
from Evaluation import evaluate
from Unet import Unet

from LabelIDs import num_classes

batch_size = 32

modelIndices = [0]

for modelIndex in modelIndices:
	model = Unet(num_classes)

	modelName = 'model_' + str(modelIndex)
	modelPath = modelName + '.ckpt'
	print('loading ' + modelPath)
	model.load_state_dict(torch.load(modelPath))

	score = evaluate(model)
	print(modelName + ' evaluated with score ' + str(score))
	writeFile("eval_" + str(modelIndex) + ".txt", str(score))