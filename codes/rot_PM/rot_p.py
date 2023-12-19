import os
from einops.layers.torch import Rearrange
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from misc import tensor2img
import cv2

imagePath = 'Demos/input'
noisePath = 'Demos/noise'
savePath = 'Demos/output'


def rot(tensor, p):
    b, _, _, _, = tensor.shape
    patchs_ro = torch.zeros_like(tensor)
    for i in range(b):
        randseed = torch.rand(1).cuda()
        if randseed <= p:
            k = torch.randint(low=0, high=5, size=(1,)).cuda()
            patchs_ro[i] = torch.rot90(tensor[i], k=int(k), dims=[1, 2])
        else:
            patchs_ro[i] = tensor[i]
    return patchs_ro


crop = transforms.Compose([
    transforms.Resize([256, 256])
])
to_patch = Rearrange('b c (h1 h) (w1 w)  -> (b h1 w1) c h w ', h1=8, w1=8)
to_entire = Rearrange('(b h1 w1) c h w  -> b c (h1 h) (w1 w) ', h1=8, w1=8)

imgNames = os.listdir(imagePath)
imgNames = sorted(imgNames)
images = []

# crop image
for imgName in imgNames:
    img = Image.open(os.path.join(imagePath, imgName))
    img_tensor = torch.from_numpy((np.array(img) / 255.0)).permute(2, 0, 1)
    img_tensor = crop(img_tensor)
    images.append(img_tensor)
images = torch.stack(images)
# crop image

# crop noise
NoiseNames = os.listdir(noisePath)
NoiseNames = sorted(NoiseNames)
Noises = []
for NoiseName in NoiseNames:
    img = Image.open(os.path.join(noisePath, NoiseName))
    img_tensor = torch.from_numpy((np.array(img) / 255.0)).permute(2, 0, 1)
    img_tensor = crop(img_tensor)
    Noises.append(img_tensor)
Noises = torch.stack(Noises)
# crop noise

# rot
M = [0.0, 0.25, 0.5, 0.75, 1.0]
P = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for m in M:
    for p in P:
        infuse_image = m * Noises + (1.0 - m) * images
        infuse_image_p = to_patch(infuse_image)  # to_patch
        infuse_image_p_rot = rot(infuse_image_p, p)  # rot patch
        infuse_image_entire = to_entire(infuse_image_p_rot)  # to_entire
        cv2.imwrite(os.path.join(savePath, "M{}_P{}_.png".format(m, p)), tensor2img(infuse_image_entire))
# rot


