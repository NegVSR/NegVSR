import cv2
import os


def image2video(src_path, sav_path, fps):
    images = []
    imageNames = os.listdir(src_path)
    imageNames = sorted(imageNames)
    dirname = os.path.split(src_path)[1]
    for imageName in imageNames:
        img = cv2.imread(os.path.join(src_path, imageName))
        images.append(img)
    w, h = img.shape[0], img.shape[1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videowrite = cv2.VideoWriter(os.path.join(sav_path, dirname + '.mp4'), fourcc, fps, (h, w))
    for i in range(len(imageNames)):
        videowrite.write(images[i])
    videowrite.release()


inputPath = ""
savePath = ""
fps = 29.7

if not os.path.exists(savePath):
    os.mkdir(savePath)

imagePaths = os.listdir(inputPath)
for imagePath in imagePaths:
    image2video(os.path.join(inputPath, imagePath), savePath, fps)
