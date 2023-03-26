import torch
from Dataset import Dataset
from Devices import device
from torchvision.models.segmentation import deeplabv3_resnet50
from torchmetrics.classification import MulticlassJaccardIndex
from Score import Score

from LabelIDs import num_classes, ignore_index, class_names

batchSize = 16

path_to_gt = "G:\\cityscapes_dataset\\val\\gts_png"
gt_to_image_path = lambda gt: gt.replace("gts_png\\gt_", "images_png\\image_")
start = 0
end = 1
shuffle = False
map_ids = True
dataset = Dataset(path_to_gt, gt_to_image_path, start, end, batchSize, shuffle, map_ids)

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

			output = model(input)['out'].detach()
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