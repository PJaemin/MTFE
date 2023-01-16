import math
import cv2
import csv
import os
import sys
import glob
import fnmatch
from IQA_pytorch import SSIM, utils
from PIL import Image
import numpy as np

# target folder
target_path = './data/train_check/'
# ground-truth folder
gt_path = './data/test_gt/'

file_list = os.listdir(target_path)


def findFile(filePath, fileName):
    ans = None
    for f_name in os.listdir(filePath):
        fileName = os.path.splitext(fileName)[0]
        if fnmatch.fnmatch(f_name, fileName + ".*"):
            ans = f_name
    if ans == None:
        print("There is no Ground-Truth matched with input file")

    return ans


def read_img(file_name):
    return cv2.imread(file_name)


def cal_PSNR(image):
    gt_file = gt_path + str(findFile(gt_path, os.path.basename(image)))
    s = read_img(image) / 255.0
    r = read_img(gt_file) / 255.0
    mse = np.mean(np.square(s - r))
    psnr = 10 * math.log10(1 / mse)

    return psnr

def PSNR1(ss,rr):
    ss = ss.cpu()
    rr = rr.cpu()
    s = ss.detach().numpy()
    r = rr.detach().numpy()

    mse = np.mean(np.square(s - r))
    psnr = 10 * math.log10(1 / mse)

    return psnr


def cal_SSIM(image):
    tg_file = image
    gt_file = gt_path + os.path.basename(tg_file)
    gt_file = gt_path + str(findFile(gt_path, os.path.basename(tg_file)))

    ref = utils.prepare_image(Image.open(gt_file).convert("RGB")).cuda()
    dist = utils.prepare_image(Image.open(tg_file).convert("RGB")).cuda()
    model = SSIM(channels=3).cuda()
    ssim = model(dist, ref, as_loss=False)
    return ssim
