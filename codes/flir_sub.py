import os
import shutil


def div23(input_path, save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    subDirs = os.listdir(input_path)
    for subDir in subDirs:
        indexs = []
        imageNames = os.listdir(os.path.join(input_path, subDir))
        imageNames = sorted(imageNames)
        indexs.append(0)
        indexs.append(round(len(imageNames) / 2))
        indexs.append(len(imageNames) - 1)
        for index in indexs:
            imagename = imageNames[index]
            init_path = os.path.join(input_path, subDir, imagename)
            tag_path = os.path.join(savePath, subDir + imagename)
            shutil.copy(init_path, tag_path)


imagesPath = 'I:\\data\\RealBasicVsr\\compare\\FLIR\\FILR_test_lr_25'
savePath = 'I:\\data\\RealBasicVsr\\compare\\FLIR\\FILR_test_lr_25_3'
div23(imagesPath, savePath)
