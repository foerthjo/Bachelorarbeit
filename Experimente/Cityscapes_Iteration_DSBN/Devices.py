import torch

cpu = torch.device('cpu')
device = torch.device('cpu')
if torch.cuda.is_available():
	print("running on gpu compute")
	device = torch.device('cuda')
else:
	print("cuda not supported. running on cpu")