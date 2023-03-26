import torch
from Dataset import Dataset, height, width
from Devices import device
from Evaluation import evaluate
import random
from torch.utils.tensorboard import SummaryWriter
from Files import writeFile
from Unet import Unet
from LabelIDs import num_classes, ignore_index
from datetime import datetime
from Cowmask import Cowmask
from ReverseCrossEntropy import reverse_cross_entropy

load_model = False

start_epoch = 0
epochs = 16
batchSize = 32
halfBatchSize = int(batchSize / 2)
assert batchSize == halfBatchSize * 2
learning_rate = 1e-4 # -4 better than -5

writer = SummaryWriter("runs/iteration_1_" + str(datetime.now()).replace(':', '_'))

model = Unet(num_classes)

if load_model:
	model_path = 'model_' + str(start_epoch - 1) + '.ckpt'
	print('loading ' + model_path)
	try:
		model.load_state_dict(torch.load(model_path))
	except:
		print('loading model failed')

model = model.to(device)

path_to_gt = "G:\\medical_decathlon_dataset\\Task01_BrainTumour\\iteration_postpro_2_1"
gt_to_image_path = lambda gt: gt.replace("iteration_postpro_2_1\\gt_", "images\\image_")
start = 0
end = 1 # labels have only been processed from 0 to .3
shuffle = True
repeat = False
map_IDs = False
dataset = Dataset(path_to_gt, gt_to_image_path, start, end, batchSize, shuffle, repeat, map_IDs)

maskScale = 32
maskBlur = 95
cowmask = Cowmask(halfBatchSize, height, width, maskScale, maskBlur)

cross_entropy = torch.nn.CrossEntropyLoss(weight=torch.tensor([1, 1, 1], dtype=torch.float32, device=device), reduction='none', ignore_index=ignore_index)
softmax = torch.nn.Softmax(dim=1)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epoch = start_epoch
while epoch < epochs:
	epochLoss = 0
	batchIndex = 0
	globalBatchIndex = epoch * dataset.batchCount
	dataset.reset()
	model.train()
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

		if random.random() < .5:
			input = input.flip([2])
			target = target.flip([1])

		cleanInput, cleanTarget = input[0:halfBatchSize], target[0:halfBatchSize]
		mixedInput, mixedTarget = input[halfBatchSize:batchSize], target[halfBatchSize:batchSize]

		with torch.no_grad():
			index = mixedTarget.view([halfBatchSize, 1, height, width])
			index.masked_fill_(index == ignore_index, 0) # confidence at pixels labeled ignore_index does not matter as the loss function ignores pixels labeled ignore_index
			confidence = torch.gather(softmax(model(mixedInput)), dim=1, index=index).view([halfBatchSize, height, width])

		optimizer.zero_grad()

		cleanOutput = model(cleanInput)
		mixedOutput = model(mixedInput)
		cleanLoss = torch.mean(cross_entropy(cleanOutput, cleanTarget))
		mixedLoss = 2 * cross_entropy(mixedOutput, mixedTarget) + reverse_cross_entropy(mixedOutput, mixedTarget, ignore_index)

		weightedLoss = mixedLoss * confidence
		weightedLoss = torch.mean(weightedLoss)

		totalLoss = cleanLoss + weightedLoss

		totalLoss.backward()
		optimizer.step()

		batchLoss = totalLoss.item()
		print('batch ' + str(batchIndex) + ' finished with loss: ' + str(batchLoss))
		writer.add_scalar('loss', batchLoss, globalBatchIndex)

		epochLoss += batchLoss
		
		batchIndex += 1
		globalBatchIndex += 1

	print('')
	modelPath = 'model_' + str(epoch) + '.ckpt'
	print('saving ' + modelPath)
	torch.save(model.state_dict(), modelPath)
	print('epoch ' + str(epoch) + ' finished with avg loss: ' + str(epochLoss / dataset.imageCount))

	model.eval()
	score = evaluate(model)
	writeFile("eval_" + str(epoch) + ".txt", str(score))
	writer.add_scalar('mean_iou', score.mean, epoch)
	writer.add_scalar('weighted_mean_iou', score.weighted_mean, epoch)
	print('')

	epoch += 1

writer.close()