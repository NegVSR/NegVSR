from PIL import Image
import numpy as np
import os
import argparse

""" 
- dataset
    -- 000
        --- 00000000.png
        --- 00000001.png
        --- 00000002.png
        ...
        --- 00000030.png
    -- 001
        --- 00000000.png
        --- 00000001.png
        --- 00000002.png
        ...
        --- 00000030.png
    -- 002
        --- 00000000.png
        --- 00000001.png
        --- 00000002.png
        ...
        --- 00000030.png
    ...
    -- xxx
-

"""

parser = argparse.ArgumentParser(description='create a dataset')
parser.add_argument('--dataset', default='Demos/FLIR_train_Group_30', type=str,
                    help='dataset path')
parser.add_argument('--ns_dir', default='Demos/Noise Sequences', type=str,
                    help='noise sequences set save dir path')
opt = parser.parse_args()


def noise_window(imgSequence, sp, max_var, min_mean, max_sq_var, max_sq_mean):
    imgs_L = []
    imgs_rgb = []
    for img_rgb in imgSequence:
        img_L = img_rgb.convert('L')

        img_rgb = np.array(img_rgb)
        img_L = np.array(img_L)

        imgs_L.append(img_L)
        imgs_rgb.append(img_rgb)

    w, h = img_L.shape
    collect_patchs = []
    collect_vars = []

    for i in range(0, w - sp, sp):
        for j in range(0, h - sp, sp):
            vars = []
            means = []
            for img_l in imgs_L:
                patch = img_l[i:i + sp, j:j + sp]
                var_global = np.var(patch)  # window var
                mean_global = np.mean(patch)  # window mean
                if var_global >= max_var or mean_global <= min_mean:
                    break
                vars.append(var_global)
                means.append(mean_global)
                if len(vars) != len(imgs_L):
                    continue
                if np.var(vars) <= max_sq_var and np.var(means) <= max_sq_mean:  # the var and mean of window sequence
                    imgs_patch = []
                    for img_rgb in imgs_rgb:
                        imgs_patch.append(img_rgb[i:i + sp, j:j + sp, :])
                    collect_patchs.append(imgs_patch)
                    collect_vars.append(vars)

    return collect_patchs, collect_vars


sp = 64  # window size
max_var = 50  # \sigma
min_mean = 0  # \mu
max_sq_var = 50  # var_{\sigma}
max_sq_mean = 50  # var_{\mu}

if not os.path.exists(opt.ns_dir):
    os.mkdir(opt.ns_dir)
subNames = os.listdir(opt.dataset)

cnt = 0
for subName in subNames:
    imgNames = os.listdir(os.path.join(opt.dataset, subName))
    imgNames.sort()
    imgSequence = []
    for imgName in imgNames:
        img_name = os.path.splitext(os.path.basename(imgName))[0]
        img_dir = os.path.join(opt.dataset, subName, imgName)
        img = Image.open(img_dir).convert('RGB')
        imgSequence.append(img)
    patchs, var_sqs = noise_window(imgSequence, sp, max_var, min_mean, max_sq_var, max_sq_mean)
    for folder_idx, patch in enumerate(patchs):
        folderPath = os.path.join(opt.ns_dir, subName + '_{}'.format(folder_idx))
        if not os.path.exists(folderPath):
            os.mkdir(folderPath)
        for idx, img in enumerate(patch):
            img_name = '{:08d}'.format(idx)
            save_path = os.path.join(folderPath,
                                     '{}_{}_{}.png'.format(subName, str(folder_idx), img_name))
            cnt += 1
            print('collect:', cnt, save_path)
            Image.fromarray(img).save(save_path)
