import torch
from Unet import Unet
from LabelIDs import num_classes
from Devices import device, cpu

cleanModel = Unet(num_classes)
augmentedModel = Unet(num_classes)

cleanModules = [m for m in cleanModel.modules() if not isinstance(m, torch.nn.BatchNorm2d)]
augmentedModules = [m for m in augmentedModel.modules() if not isinstance(m, torch.nn.BatchNorm2d)]

i = 0
while i < len(cleanModules):
	if hasattr(augmentedModules[i], 'weight'):
		augmentedModules[i].weight = cleanModules[i].weight
	if hasattr(augmentedModules[i], 'weights'):
		augmentedModules[i].weights = cleanModules[i].weights
	i += 1