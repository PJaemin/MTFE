import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
from Model import Image_network
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time
from matplotlib import pyplot as plt
import cv2
from Metrics import cal_PSNR
from unet_model import UNet
import torch.nn.functional as F

def get_hist(file_name):
    src = cv2.imread(file_name)
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    hist_s = np.zeros((3, 256))

    for (j, color) in enumerate(("red", "green", "blue")):
        S = src[..., j]
        hist_s[j, ...], _ = np.histogram(S.flatten(), 256, [0, 256])
        hist_s[j, ...] = hist_s[j, ...] / np.sum(hist_s[j, ...])

    hist_s = torch.from_numpy(hist_s).float()

    return hist_s


def lowlight(image_path):
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 0번 GPU에 메모리 할당
    data_lowlight = Image.open(image)
    data_lowlight = (np.asarray(data_lowlight) / 255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.cuda().unsqueeze(0)
    hist = get_hist(image)
    hist = hist.cuda().unsqueeze(0)
    Imgnet = Image_network()
    Imgnet = Imgnet.cuda()
    Imgnet.eval()
    Imgnet.load_state_dict(torch.load('models/Img_final.pth'))

    enhanced_img, vec, wm, xy = Imgnet(data_lowlight, hist)

    result_path = image.replace('test_data', 'analysis/result')
    plot_path = image.replace('test_data', 'analysis/test_plots')
    wm_path1 = image.replace('test_data/LOL', 'analysis/test_weightmap1')
    wm_path2 = image.replace('test_data/LOL', 'analysis/test_weightmap2')
    wm_path3 = image.replace('test_data/LOL', 'analysis/test_weightmap3')
    xy_path1 = image.replace('test_data/LOL', 'analysis/test_output1')
    xy_path2 = image.replace('test_data/LOL', 'analysis/test_output2')
    xy_path3 = image.replace('test_data/LOL', 'analysis/test_output3')

    if not os.path.exists(result_path.replace('/' + result_path.split("/")[-1], '')):
        os.makedirs(result_path.replace('/' + result_path.split("/")[-1], ''))
    if not os.path.exists(plot_path.replace('/' + plot_path.split("/")[-1], '')):
        os.makedirs(plot_path.replace('/' + plot_path.split("/")[-1], ''))
    if not os.path.exists(wm_path1.replace('/' + wm_path1.split("/")[-1], '')):
        os.makedirs(wm_path1.replace('/' + wm_path1.split("/")[-1], ''))
    if not os.path.exists(wm_path2.replace('/' + wm_path2.split("/")[-1], '')):
        os.makedirs(wm_path2.replace('/' + wm_path2.split("/")[-1], ''))
    if not os.path.exists(wm_path3.replace('/' + wm_path3.split("/")[-1], '')):
        os.makedirs(wm_path3.replace('/' + wm_path3.split("/")[-1], ''))
    if not os.path.exists(xy_path1.replace('/' + xy_path1.split("/")[-1], '')):
        os.makedirs(xy_path1.replace('/' + xy_path1.split("/")[-1], ''))
    if not os.path.exists(xy_path2.replace('/' + xy_path2.split("/")[-1], '')):
        os.makedirs(xy_path2.replace('/' + xy_path2.split("/")[-1], ''))
    if not os.path.exists(xy_path3.replace('/' + xy_path3.split("/")[-1], '')):
        os.makedirs(xy_path3.replace('/' + xy_path3.split("/")[-1], ''))

    torchvision.utils.save_image(enhanced_img, result_path)
    torchvision.utils.save_image(wm[0], wm_path1)
    torchvision.utils.save_image(wm[1], wm_path2)
    torchvision.utils.save_image(wm[2], wm_path3)
    torchvision.utils.save_image(xy[0], xy_path1)
    torchvision.utils.save_image(xy[1], xy_path2)
    torchvision.utils.save_image(xy[2], xy_path3)

    if not os.path.exists(plot_path.replace('/' + plot_path.split("/")[-1], '')):
        os.makedirs(plot_path.replace('/' + plot_path.split("/")[-1], ''))

    vec1 = vec[0]
    vec2 = vec[1]
    vec3 = vec[2]

    (fig, axs) = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    vec1 = vec1.squeeze(0)
    vec1 = vec1.cpu().detach().numpy()
    vec1 = vec1 * 255
    r1 = vec1[0, ...]
    g1 = vec1[1, ...]
    b1 = vec1[2, ...]
    axs[0].plot(b1, color='b')
    axs[0].plot(g1, color='g')
    axs[0].plot(r1, color='r')

    vec2 = vec2.squeeze(0)
    vec2 = vec2.cpu().detach().numpy()
    vec2 = vec2 * 255
    r2 = vec2[0, ...]
    g2 = vec2[1, ...]
    b2 = vec2[2, ...]
    axs[1].plot(b2, color='b')
    axs[1].plot(g2, color='g')
    axs[1].plot(r2, color='r')


    vec3 = vec3.squeeze(0)
    vec3 = vec3.cpu().detach().numpy()
    vec3 = vec3 * 255
    r3 = vec3[0, ...]
    g3 = vec3[1, ...]
    b3 = vec3[2, ...]
    axs[2].plot(b3, color='b')
    axs[2].plot(g3, color='g')
    axs[2].plot(r3, color='r')


    axs[0].set_xlim([0, 255])
    axs[0].set_ylim([0, 255])
    axs[1].set_xlim([0, 255])
    axs[1].set_ylim([0, 255])
    axs[2].set_xlim([0, 255])
    axs[2].set_ylim([0, 255])

    plt.tight_layout()
    plt.draw()
    plt.savefig(plot_path)





# 파일이 import에 의해서가 아닌 interpreter에 의해서 호출될때만 실행 가능
if __name__ == '__main__':
    # test_images
    with torch.no_grad():  # gradient 연산 옵션을 끔. 이 내부 컨텍스트 텐서들은 requires_grad=False 되어 메모리사용 아낌
        filePath = 'data/test_data/'  # test dataset path

        file_list = os.listdir(filePath)  # os.listdir은 디렉토리내에 모든 파일과 디렉토리 리스트를 리턴함

        best = 0
        ep = 0
        sum_psnr=0
        for file_name in file_list:  # DCIM,LIME까지
            test_list = glob.glob(filePath + file_name + "/*")  # filePath+file_name에 해당되는 모든 파일
            n_of_files = len(test_list)
            for image in test_list:
                # image = image
                # print(image)
                lowlight(image)
                print('[Done] ' + str(image))
                # result_path = image.replace('test_data', 'result')
                # img_psnr = cal_PSNR(result_path)
                # print(str(image) + '\tPSNR: '+str(img_psnr))
                # sum_psnr += img_psnr

        # avg_psnr = sum_psnr / n_of_files
        # print('Avg_PSNR: %.3f' % (avg_psnr))

        # for file_name in file_list:  # DCIM,LIME까지
        #     test_list = glob.glob(filePath + file_name + "/*")  # filePath+file_name에 해당되는 모든 파일
        #     for image in test_list:
        #         # image = image
        #         print(image)
        #         lowlight(image)
