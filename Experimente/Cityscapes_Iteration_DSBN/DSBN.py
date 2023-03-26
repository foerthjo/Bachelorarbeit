import torch
from torchvision.models.segmentation import deeplabv3_resnet50
from LabelIDs import num_classes
from Devices import device, cpu

model_weights = deeplabv3_resnet50(weights="DEFAULT").to(device)
cleanModel = deeplabv3_resnet50(weights=None, num_classes=num_classes).to(device)
cleanModel.backbone = model_weights.backbone
model_weights = None

model_weights = deeplabv3_resnet50(weights="DEFAULT").to(device)
augmentedModel = deeplabv3_resnet50(weights=None, num_classes=num_classes).to(device)
augmentedModel.backbone = model_weights.backbone
model_weights = None

cleanModules = [m for m in cleanModel.modules() if not isinstance(m, torch.nn.BatchNorm2d)]
augmentedModules = [m for m in augmentedModel.modules() if not isinstance(m, torch.nn.BatchNorm2d)]

i = 0
while i < len(cleanModules):
	if hasattr(augmentedModules[i], 'weight'):
		augmentedModules[i].weight = cleanModules[i].weight
	if hasattr(augmentedModules[i], 'weights'):
		augmentedModules[i].weights = cleanModules[i].weights
	i += 1