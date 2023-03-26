import torch
from Dataset import Dataset
from Devices import device
from Evaluation import evaluate
import random
from torch.utils.tensorboard import SummaryWriter
from Files import writeFile
from Unet import Unet
from datetime import datetime

from LabelIDs import num_classes, ignore_index

load_model = False

start_epoch = 0
epochs = 16
batchSize = 32
learning_rate = 1e-4 # -4 better than -5

writer = SummaryWriter("runs/iteration_postpro_tumor_2_1" + str(datetime.now()).replace(':', '_'))

model = Unet(num_classes)

if load_model:
	print('loading model')
	try:
		model.load_state_dict(torch.load("model.ckpt"))
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

loss_fn = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=ignore_index)
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
		input, segmentation = data

		optimizer.zero_grad()

		if random.random() < .5:
			input = input.flip([3])
			segmentation = segmentation.flip([2])

		if random.random() < .5:
			input = input.flip([2])
			segmentation = segmentation.flip([1])

		output = model(input)
		loss = loss_fn(output, segmentation)

		loss.backward()
		optimizer.step()

		batchLoss = loss.item()
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