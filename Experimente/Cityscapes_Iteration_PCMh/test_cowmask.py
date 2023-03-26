import torch
from Dataset import Dataset, height, width
from Cowmask import Cowmask
from torchvision.io import write_png
from Devices import device, cpu

batchSize = 16

scale = 16
blur = 95
cowmask = Cowmask(8, height, width, scale, blur)

path_to_gt = "G:\\cityscapes_dataset\\train\\iteration_postpro_1"
gt_to_image_path = lambda gt: gt.replace("iteration_postpro_1\\labels_", "images_png\\image_")
start = 0
end = 1
shuffle = True
map_ids = False
dataset = Dataset(path_to_gt, gt_to_image_path, start, end, batchSize, shuffle, map_ids)

images, labels = dataset.loadBatch()

img, lbl, conf = cowmask.mix((images[0:8], labels[0:8], labels[0:8]), (images[8:16], labels[8:16], labels[8:16]))

write_png((img[1] * 255).type(torch.uint8).to(cpu), 'test3_img1.png', 9)
write_png(lbl[1].type(torch.uint8).view([1, height, width]).to(cpu), 'test3_label1.png', 9)
write_png(conf[1].type(torch.uint8).view([1, height, width]).to(cpu), 'test3_conf1.png', 9)