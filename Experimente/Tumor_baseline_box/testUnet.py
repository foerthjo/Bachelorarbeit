from Unet import Unet
import torch

model = Unet(4)

print(model)

batch = torch.empty((8, 4, 240, 240))
output = model(batch)
print(output.shape)