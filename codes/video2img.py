import cv2
import os


def video2img(video_path, output_path):
    interval = 10  # 设置每间隔10帧取一张图片
    num = 1  # 计数单位
    vid = cv2.VideoCapture(video_path)
    while vid.isOpened():
        is_read, frame = vid.read()
        if is_read:
            if num % interval == 1:  # 每间隔10帧取一张图片
                file_name = '%08d' % num
                cv2.imwrite(os.path.join(output_path, file_name + '.png'), frame)  # 写入文件
                print(file_name)
                cv2.waitKey(1)
            num += 1
        else:
            break
    # while vid.isOpened():
    #     is_read, frame = vid.read()
    #     if is_read:
    #         file_name = '%08d' % num
    #         cv2.imwrite(os.path.join(output_path, file_name + '.png'), frame)  # 写入文件
    #         print(file_name)
    #         cv2.waitKey(1)
    #         num += 1
    #     else:
    #         break


dir = ''
save_dir = ''

# 保存目录不存在则新建保存目录
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


videoNames = os.listdir(dir)
for videoName in videoNames:
    videoPath = os.path.join(dir, videoName)
    video, _ = videoName.split('.')
    dir_sub = os.path.join(save_dir, video)
    if not os.path.exists(dir_sub):
        os.mkdir(dir_sub)
    video2img(videoPath, dir_sub)


