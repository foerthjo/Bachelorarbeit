import torch
from Dataset import Dataset
from Devices import device
from Score import Score

from LabelIDs import num_classes, class_names

batchSize = 32

path_to_gt = "G:\\medical_decathlon_dataset\\Task01_BrainTumour\\gts_png"
gt_to_image_path = lambda gt: gt.replace("gts_png\\gt_", "images\\image_")
start = .3
end = .4
shuffle = False
repeat = False
map_IDs = True
dataset = Dataset(path_to_gt, gt_to_image_path, start, end, batchSize, shuffle, repeat, map_IDs)

def evaluate(model):
	score = Score(num_classes, True, class_names)

	with torch.no_grad():
		print('evaluating')
		model = model.to(device)
		model.eval()
		dataset.resetIndex()
		
		while True:
			batch = dataset.loadBatch()
			if batch == None:
				break

			input, segmentation = batch
			localBatchSize, channels, height, width = input.shape

			output = model(input).detach()
			prediction = torch.argmax(output, dim=1, keepdim=False)

			tp = [0] * num_classes
			fp = [0] * num_classes
			fn = [0] * num_classes
			for i in range(num_classes):
				gt = segmentation.eq(i).type(torch.int)
				pred = prediction.eq(i).type(torch.int)
				tp[i] = torch.sum(gt * pred).item()
				fp[i] = torch.sum(pred * (1 - gt)).item()
				fn[i] = torch.sum(gt * (1 - pred)).item()

			score.add(tp, fp, fn)
			
			print('batch finished with iou: ' + score.quickSummary())

	score.calculate_ious()
	return score