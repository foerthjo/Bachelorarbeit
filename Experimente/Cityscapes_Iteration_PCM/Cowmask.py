import torchvision
from torchvision.io import write_png
import torch
from Devices import device, cpu

class Cowmask:
	def __init__(this, batchSize, height, width, scale, blur):
		this.scale = scale
		this.batchSize = batchSize
		this.halfBatchSize = int(batchSize / 2)
		assert this.halfBatchSize * 2 == this.batchSize
		this.height = height
		this.width = width
		this.upscale = torchvision.transforms.Resize([height, width], interpolation=torchvision.transforms.InterpolationMode.BILINEAR, max_size=None, antialias=True)
		this.blur = torchvision.transforms.GaussianBlur(blur, sigma=(0.1, 2.0))
	
	def newMask(this):
		noise = torch.rand([1, int(this.height / this.scale), int(this.width / this.scale)], device=device, dtype=torch.float32)
		noise = this.upscale(noise)
		noise = this.blur(noise)
		noise = (noise > .5).type(torch.uint8)
		imageMask = noise.view([1, 1, this.height, this.width]).expand([this.halfBatchSize, 3, this.height, this.width])
		labelMask = noise.view([1, this.height, this.width]).expand([this.halfBatchSize, this.height, this.width])
		return imageMask, labelMask
	
	def mix(this, images, labels):
		images1 = images[0:this.halfBatchSize]
		images2 = images[this.halfBatchSize:this.batchSize]
		labels1 = labels[0:this.halfBatchSize]
		labels2 = labels[this.halfBatchSize:this.batchSize]
		
		imageMask, labelMask = this.newMask()
		imageMaskInv = 1 - imageMask
		labelMaskInv = 1 - labelMask
		images = torch.cat([images1 * imageMask + images2 * imageMaskInv, images1 * imageMaskInv + images2 * imageMask], dim=0)
		labels = torch.cat([labels1 * labelMask + labels2 * labelMaskInv, labels1 * labelMaskInv + labels2 * labelMask], dim=0)

		return images, labels