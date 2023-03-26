import torch
import torchvision

class ConvBlock(torch.nn.Module):
	def __init__(this, in_channels, featureCount):
		super().__init__()

		this.block = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=in_channels, out_channels=featureCount, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], groups=1, bias=False, dilation=1, padding_mode='zeros'),
			torch.nn.BatchNorm2d(featureCount),
			# torch.nn.LeakyReLU(negative_slope=0.01, inplace=False),
			torch.nn.ReLU(inplace=True),

			torch.nn.Conv2d(in_channels=featureCount, out_channels=featureCount, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], groups=1, bias=False, dilation=1, padding_mode='zeros'),
			torch.nn.BatchNorm2d(featureCount),
			torch.nn.ReLU(inplace=True),
		)

		this.pooling = torch.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, return_indices=False, ceil_mode=False)
	
	def forward(this, input):
		intermediate = this.block(input)
		input = this.pooling(intermediate)
		return (input, intermediate)

class Transformation(torch.nn.Module):
	def __init__(this, in_channels, featureCount):
		super().__init__()

		this.block = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=in_channels, out_channels=featureCount, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], groups=1, bias=False, dilation=1, padding_mode='zeros'),
			torch.nn.BatchNorm2d(featureCount),
			torch.nn.ReLU(inplace=True),

			torch.nn.Conv2d(in_channels=featureCount, out_channels=featureCount, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], groups=1, bias=False, dilation=1, padding_mode='zeros'),
			torch.nn.BatchNorm2d(featureCount),
			torch.nn.ReLU(inplace=True),
		)
	
	def forward(this, input):
		return this.block(input)

class Upsampling(torch.nn.Module):
	def __init__(this, in_channels, featureCount):
		super().__init__()

		this.upsampling = torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=featureCount, kernel_size=[2, 2], stride=[2, 2], padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')
		
		this.conv = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=featureCount*2, out_channels=featureCount, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], groups=1, bias=False, dilation=1, padding_mode='zeros'),
			torch.nn.BatchNorm2d(featureCount),
			torch.nn.ReLU(inplace=True),

			torch.nn.Conv2d(in_channels=featureCount, out_channels=featureCount, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], groups=1, bias=False, dilation=1, padding_mode='zeros'),
			torch.nn.BatchNorm2d(featureCount),
			torch.nn.ReLU(inplace=True),
		)
	
	def forward(this, input, intermediate):
		# input = torchvision.transforms.functional.resize(input, intermediate.shape[2:], interpolation=torchvision.transforms.InterpolationMode.BILINEAR, max_size=None, antialias=None)
		input = this.upsampling(input)

		if input.shape != intermediate.shape:
			torchvision.transforms.functional.resize(input, intermediate.shape[2:], interpolation=torchvision.transforms.InterpolationMode.BILINEAR, max_size=None, antialias=None)

		input = torch.cat((intermediate, input), dim=1)
		input = this.conv(input)
		return input

class Unet(torch.nn.Module):
	def __init__(this, num_classes):
		super().__init__()
		
		this.conv1 = ConvBlock(4, 64)
		this.conv2 = ConvBlock(64, 128)
		this.conv3 = ConvBlock(128, 256)
		this.conv4 = ConvBlock(256, 512)

		this.transformation = Transformation(512, 1024)

		this.upsample1 = Upsampling(1024, 512)
		this.upsample2 = Upsampling(512, 256)
		this.upsample3 = Upsampling(256, 128)
		this.upsample4 = Upsampling(128, 64)

		this.classifier = torch.nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], groups=1, bias=False, dilation=1, padding_mode='zeros')

	def forward(this, input):
		input, intermediate1 = this.conv1(input)
		input, intermediate2 = this.conv2(input)
		input, intermediate3 = this.conv3(input)
		input, intermediate4 = this.conv4(input)

		input = this.transformation(input)

		input = this.upsample1(input, intermediate4)
		input = this.upsample2(input, intermediate3)
		input = this.upsample3(input, intermediate2)
		input = this.upsample4(input, intermediate1)

		input = this.classifier(input)

		return input