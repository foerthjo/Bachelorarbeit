import torch
from LabelIDs import num_classes

epsilon = 1e-8

softmax = torch.nn.Softmax(dim=1)

def reverse_cross_entropy(predictions, labels, ignore_index):
	batchSize, num_classes, height, width = predictions.shape

	predictions = softmax(predictions) + epsilon

	if (torch.min(predictions) == 0):
		print('warning: reverse cross entropy is zero')
		return 0
	
	ignore_mask = labels == ignore_index
	labels = torch.nn.functional.one_hot(labels.masked_fill(ignore_mask, 0), num_classes=num_classes)
	labels = labels.moveaxis(3, 1)
	predictions = torch.log(predictions)
	predictions = torch.sum(predictions * labels, dim=1, keepdim=False)
	predictions = predictions * ignore_mask.type(torch.uint8)
	return predictions.view(batchSize, height, width) * -1