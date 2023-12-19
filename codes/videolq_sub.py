import os
import shutil

Divpath = "/media/abc/DA18EBFA09C1B27D/Song/test/Output"
Path3 = "/media/abc/DA18EBFA09C1B27D/Song/test/Output3"
Allpath = "/home/abc/SongFlies/data/RealBasicVsrDriveResult/experiments_neg_MixupRot_VideoLQnoise_weight05_patchSize4_layer10_len30_All"


def Div2All(path, topath):
    for root, dir, file in os.walk(path, topdown="False"):
        if len(file) != 0:
            for filename in file:
                shutil.copyfile(os.path.join(root, filename), os.path.join(topath, '000' + root[-3:] + filename[-6:]))


def Div23(path, topath):
    for root, dir, file in os.walk(path, topdown="False"):
        if len(file) != 0:
            for filename in file:
                if filename == '00000000.png' or filename == '00000050.png' or filename == '00000099.png':
                    shutil.copyfile(os.path.join(root, filename),
                                    os.path.join(topath, '000' + root[-3:] + filename[-6:]))
                elif root[-3:] == '030' and (filename == '00000020.png' or filename == '00000040.png'):
                    shutil.copyfile(os.path.join(root, filename),
                                    os.path.join(topath, '000' + root[-3:] + filename[-6:]))
                elif root[-3:] == '031' and (filename == '00000017.png' or filename == '00000033.png'):
                    shutil.copyfile(os.path.join(root, filename),
                                    os.path.join(topath, '000' + root[-3:] + filename[-6:]))
                elif root[-3:] == '032' and (filename == '00000024.png' or filename == '00000048.png'):
                    shutil.copyfile(os.path.join(root, filename),
                                    os.path.join(topath, '000' + root[-3:] + filename[-6:]))
                elif root[-3:] == '033' and (filename == '00000023.png' or filename == '00000046.png'):
                    shutil.copyfile(os.path.join(root, filename),
                                    os.path.join(topath, '000' + root[-3:] + filename[-6:]))


def Div23_div(path, topath):
    for root, dir, file in os.walk(path, topdown="False"):

        if len(file) != 0:
            for filename in file:
                if not os.path.exists(os.path.join(topath, root[-3:])):
                    os.mkdir(os.path.join(topath, root[-3:]))
                if filename == '00000000.png' or filename == '00000050.png' or filename == '00000099.png':
                    shutil.copyfile(os.path.join(root, filename),
                                    os.path.join(topath, root[-3:], '000' + root[-3:] + filename[-6:]))
                elif root[-3:] == '030' and (filename == '00000020.png' or filename == '00000040.png'):
                    shutil.copyfile(os.path.join(root, filename),
                                    os.path.join(topath, root[-3:], '000' + root[-3:] + filename[-6:]))
                elif root[-3:] == '031' and (filename == '00000017.png' or filename == '00000033.png'):
                    shutil.copyfile(os.path.join(root, filename),
                                    os.path.join(topath, root[-3:], '000' + root[-3:] + filename[-6:]))
                elif root[-3:] == '032' and (filename == '00000024.png' or filename == '00000048.png'):
                    shutil.copyfile(os.path.join(root, filename),
                                    os.path.join(topath, root[-3:], '000' + root[-3:] + filename[-6:]))
                elif root[-3:] == '033' and (filename == '00000023.png' or filename == '00000046.png'):
                    shutil.copyfile(os.path.join(root, filename),
                                    os.path.join(topath, root[-3:], '000' + root[-3:] + filename[-6:]))


if not os.path.exists(Path3):
    os.mkdir(Path3)

# if not os.path.exists(Allpath):
#     os.mkdir(Allpath)

# Div2All(Divpath, Allpath)
Div23(Divpath, Path3)
