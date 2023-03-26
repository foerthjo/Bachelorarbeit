import torch

ignore_index = 255

labels = [
	('background', 0, 0),
	('edema', 1, 1),
	('necrotic tumor', 2, 2),
	('enhancing tumor', 3, 2),
]

num_classes = 3
class_names = [
	'background', 'edema', 'tumor'
]

labelIDsByName = {}
trainIDsByLabelID = list(range(35))
for (name, id, trainID) in labels:
	labelIDsByName[name] = id
	trainIDsByLabelID[id] = trainID

def mapLabelIDsToTrain(tensor):
	mapped = tensor * 1
	for (name, id, trainID) in labels:
		mask = torch.eq(tensor, id)
		mapped.masked_fill_(mask, trainID)
	return mapped