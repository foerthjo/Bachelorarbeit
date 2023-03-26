import torch
from torch import nn
from torchvision.io import read_image
from torchvision.io import write_jpeg
import torchvision.transforms as transforms
from Dataset import Dataset
from Devices import device
from Evaluation import evaluate
from torchvision.models.segmentation import deeplabv3_resnet50
import random
from torch.utils.tensorboard import SummaryWriter
from Files import writeFile
from LabelIDs import num_classes, ignore_index

load_model = False

start_epoch = 0
epochs = 32
batchSize = 16
learning_rate = 1e-4 # -4 better than -5

writer = SummaryWriter("runs/boxes_i_0")

model_weights = deeplabv3_resnet50(weights="DEFAULT").to(device)
model = deeplabv3_resnet50(weights=None, num_classes=num_classes).to(device)
model.backbone = model_weights.backbone
model_weights = None

if load_model:
	model_path = 'model_' + str(start_epoch - 1) + '.ckpt'
	print('loading ' + model_path)
	try:
		model.load_state_dict(torch.load(model_path))
	except:
		print('loading model failed')

model = model.to(device)

path_to_gt = "G:\\cityscapes_dataset\\train\\boxes_i"
gt_to_image_path = lambda gt: gt.replace("boxes_i\\boxes_", "images_png\\image_")
start = 0
end = 1
shuffle = True
map_ids = False
dataset = Dataset(path_to_gt, gt_to_image_path, start, end, batchSize, shuffle, map_ids)

loss_fn = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=ignore_index)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epoch = start_epoch
while epoch < epochs:
	batchIndex = 0
	globalBatchIndex = epoch * dataset.batchCount
	dataset.reset()
	model.train()
	while True:
		data = dataset.loadBatch()
		if data == None:
			break
		input, target = data

		optimizer.zero_grad()

		if random.random() < .5:
			input = input.flip([3])
			target = target.flip([2])

		output = model(input)['out']
		loss = loss_fn(output, target)

		loss.backward()
		optimizer.step()

		batchLoss = loss.item()
		print('batch ' + str(batchIndex) + ' finished with loss: ' + str(batchLoss))
		writer.add_scalar('loss', batchLoss, globalBatchIndex)
		
		batchIndex += 1
		globalBatchIndex += 1

	print('')
	modelPath = 'model_' + str(epoch) + '.ckpt'
	print('saving ' + modelPath)
	torch.save(model.state_dict(), modelPath)

	model.eval()
	score = evaluate(model)
	writeFile("eval_" + str(epoch) + ".txt", str(score))
	writer.add_scalar('mean_iou', score.mean, epoch)
	writer.add_scalar('weighted_mean_iou', score.weighted_mean, epoch)
	print('')

	epoch += 1

writer.close()