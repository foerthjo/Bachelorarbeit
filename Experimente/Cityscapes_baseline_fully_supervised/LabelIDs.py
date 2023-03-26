import torch

ignore_index = 255

'''
labels = [
	('unlabeled', 0, 255),
	('ego vehicle', 1, 255),
	('rectification border', 2, 255),
	('out of roi', 3, 255),
	('static', 4, 255),
	('dynamic', 5, 255),
	('ground', 6, 255),
	('road', 7, 0),
	('sidewalk', 8, 1),
	('parking', 9, 255),
	('rail track', 10, 255),
	('building', 11, 2),
	('wall', 12, 3),
	('fence', 13, 4),
	('guard rail', 14, 255),
	('bridge', 15, 255),
	('tunnel', 16, 255),
	('pole', 17, 5),
	('polegroup', 18, 255),
	('traffic light', 19, 6),
	('traffic sign', 20, 7),
	('vegetation', 21, 8),
	('terrain', 22, 9),
	('sky', 23, 10),
	('person', 24, 11),
	('rider', 25, 12),
	('car', 26, 13),
	('truck', 27, 14),
	('bus', 28, 15),
	('caravan', 29, 255),
	('trailer', 30, 255),
	('train', 31, 16),
	('motorcycle', 32, 17),
	('bicycle', 33, 18),
	('license plate', 34, 255),
]
num_classes = 19
'''

labels = [
	('unlabeled', 0, 0),
	('ego vehicle', 1, 0),
	('rectification border', 2, 0),
	('out of roi', 3, 0),
	('static', 4, 0),
	('dynamic', 5, 0),
	('ground', 6, 0),
	('road', 7, 0),
	('sidewalk', 8, 0),
	('parking', 9, 0),
	('rail track', 10, 0),
	('building', 11, 0),
	('wall', 12, 0),
	('fence', 13, 0),
	('guard rail', 14, 0),
	('bridge', 15, 0),
	('tunnel', 16, 0),
	('pole', 17, 0),
	('polegroup', 18, 0),
	('traffic light', 19, 1),
	('traffic sign', 20, 2),
	('vegetation', 21, 0),
	('terrain', 22, 0),
	('sky', 23, 0),
	('person', 24, 3),
	('rider', 25, 3),
	('car', 26, 4),
	('truck', 27, 4),
	('bus', 28, 4),
	('caravan', 29, 4),
	('trailer', 30, 4),
	('train', 31, 0),
	('motorcycle', 32, 5),
	('bicycle', 33, 6),
	('license plate', 34, 0),
]

num_classes = 7
class_names = [
	'background', 'traffic light', 'traffic sign', 'person', '4-wheel vehicle', 'motorcycle', 'bycicle'
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