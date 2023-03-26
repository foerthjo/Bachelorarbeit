import torch
from torch import nn
from torchvision.io import read_image
from torchvision.io import write_jpeg
import torchvision.transforms as transforms
from Dataset import Dataset, height, width
from Devices import device
from Evaluation import evaluate
from torchvision.models.segmentation import deeplabv3_resnet50
import random
from torch.utils.tensorboard import SummaryWriter
from Files import writeFile
from LabelIDs import num_classes, ignore_index
from Cowmask import Cowmask
from ReverseCrossEntropy import reverse_cross_entropy
from datetime import datetime
from DSBN import cleanModel, augmentedModel

load_models = False

start_epoch = 0
epochs = 32
batchSize = 16
halfBatchSize = int(batchSize / 2)
assert batchSize == halfBatchSize * 2
learning_rate = 1e-4 # 1e-4 better than 1e-5

writer = SummaryWriter("runs/DSBN_" + str(datetime.now()).replace(':', '_'))

if load_models:
	print('loading models')
	try:
		cleanModel.load_state_dict(torch.load('cleanModel_' + str(start_epoch - 1) + '.ckpt'))
		augmentedModel.load_state_dict(torch.load('augmentedModel_' + str(start_epoch - 1) + '.ckpt'))
	except:
		print('loading models failed')

path_to_gt = "G:\\cityscapes_dataset\\train\\iteration_postpro_1"
gt_to_image_path = lambda gt: gt.replace("iteration_postpro_1\\labels_", "images_png\\image_")
start = 0
end = 1
shuffle = True
map_ids = False
dataset = Dataset(path_to_gt, gt_to_image_path, start, end, batchSize, shuffle, map_ids)

maskScale = 16
maskBlur = 95
cowmask = Cowmask(halfBatchSize, height, width, maskScale, maskBlur)

cross_entropy = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)
softmax = torch.nn.Softmax(dim=1)
cleanOptimizer = torch.optim.Adam(cleanModel.parameters(), lr=learning_rate)
augmentedOptimizer = torch.optim.Adam(augmentedModel.parameters(), lr=learning_rate)

epoch = start_epoch
while epoch < epochs:
	batchIndex = 0
	globalBatchIndex = epoch * dataset.batchCount
	dataset.reset()
	cleanModel.train()
	augmentedModel.train()
	while True:
		data = dataset.loadBatch()
		if data == None:
			break
		
		input, target = data
		if input.shape[0] != batchSize or target.shape[0] != batchSize:
			break
		
		if random.random() < .5:
			input = input.flip([3])
			target = target.flip([2])

		cleanInput, cleanTarget = input[0:halfBatchSize], target[0:halfBatchSize]
		mixedInput, mixedTarget = input[halfBatchSize:batchSize], target[halfBatchSize:batchSize]

		with torch.no_grad():
			index = mixedTarget.view([halfBatchSize, 1, height, width])
			index.masked_fill_(index == ignore_index, 0) # confidence at pixels labeled ignore_index does not matter as the loss function ignores pixels labeled ignore_index
			confidence = torch.gather(softmax(cleanModel(mixedInput)['out']), dim=1, index=index).view([halfBatchSize, height, width])

		mixedInput, mixedTarget, confidence = cowmask.mix(mixedInput, mixedTarget, confidence)

		cleanOptimizer.zero_grad()
		augmentedOptimizer.zero_grad()

		cleanOutput = cleanModel(cleanInput)['out']
		mixedOutput = augmentedModel(mixedInput)['out']
		cleanLoss = torch.mean(cross_entropy(cleanOutput, cleanTarget))
		mixedLoss = 2 * cross_entropy(mixedOutput, mixedTarget) + reverse_cross_entropy(mixedOutput, mixedTarget, ignore_index)
		trainProgress = torch.exp(-(cleanLoss.detach() * 2))

		weightedLoss = mixedLoss * confidence
		weightedLoss = torch.mean(weightedLoss) * trainProgress

		totalLoss = cleanLoss + weightedLoss

		totalLoss.backward()
		cleanOptimizer.step()
		augmentedOptimizer.step()

		batchLoss = totalLoss.item()
		print('batch ' + str(batchIndex) + ' finished with loss: ' + str(batchLoss) + ', training progress: ' + str(trainProgress.item()))
		writer.add_scalar('loss', batchLoss, globalBatchIndex)
		writer.add_scalar('training_progress', trainProgress.item(), globalBatchIndex)
		
		batchIndex += 1
		globalBatchIndex += 1

	print('')
	print('saving models')
	torch.save(cleanModel.state_dict(), 'cleanModel_' + str(epoch) + '.ckpt')
	torch.save(augmentedModel.state_dict(), 'augmentedModel_' + str(epoch) + '.ckpt')

	cleanModel.eval()
	score = evaluate(augmentedModel)
	writeFile("eval_" + str(epoch) + ".txt", str(score))
	writer.add_scalar('mean_iou', score.mean, epoch)
	writer.add_scalar('weighted_mean_iou', score.weighted_mean, epoch)
	print('')

	epoch += 1

writer.close()