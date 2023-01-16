import torch
import shutil
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
from Model import Image_network
import Myloss
from torch.utils.tensorboard import SummaryWriter
import glob
from Metrics import cal_PSNR
from PIL import Image
import numpy as np
import cv2
from distutils.dir_util import copy_tree
from matplotlib import pyplot as plt

writer = SummaryWriter()
GPU_NUM = 1


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


def eval(model, save_plot=False):
    filePath = 'data/test_data/'  # test dataset path

    file_list = os.listdir(filePath)  # os.listdir은 디렉토리내에 모든 파일과 디렉토리 리스트를 리턴함
    sum_psnr = 0
    n_of_files = 0

    for file_name in file_list:  # DCIM,LIME까지
        test_list = glob.glob(filePath + file_name + "/*")  # filePath+file_name에 해당되는 모든 파일
        n_of_files = len(test_list)
        for image in test_list:
            data_lowlight = Image.open(image)
            data_lowlight = (np.asarray(data_lowlight) / 255.0)
            data_lowlight = torch.from_numpy(data_lowlight).float()
            # data_lowlight = data_lowlight * 2.0 - 1.0
            data_lowlight = data_lowlight.permute(2, 0, 1)
            data_lowlight = data_lowlight.cuda().unsqueeze(0)
            hist = get_hist(image)
            hist = hist.cuda().unsqueeze(0)
            # hist = hist * 2.0 - 1.0
            Imgnet = Image_network()
            Imgnet = Imgnet.cuda()
            Imgnet.eval()
            Imgnet.load_state_dict(torch.load('models/Img_' + model + '.pth'))

            enhanced_img, vec, wm, xy = Imgnet(data_lowlight, hist)
            # enhanced_img = enhanced_img * 0.5 + 0.5

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

            if save_plot == True:
                if not os.path.exists(plot_path.replace('/' + plot_path.split("/")[-1], '')):
                    os.makedirs(plot_path.replace('/' + plot_path.split("/")[-1], ''))

                vec1 = vec[0]
                vec2 = vec[1]
                vec3 = vec[2]

                (fig, axs) = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
                vec1 = vec1.squeeze(0)
                # vec1 = vec1 * 0.5 + 0.5
                vec1 = vec1.cpu().detach().numpy()
                vec1 = vec1 * 255
                r1 = vec1[0, ...]
                g1 = vec1[1, ...]
                b1 = vec1[2, ...]
                axs[0][0].plot(r1, color='r')
                axs[0][1].plot(g1, color='g')
                axs[0][2].plot(b1, color='b')

                vec2 = vec2.squeeze(0)
                # vec2 = vec2 * 0.5 + 0.5
                vec2 = vec2.cpu().detach().numpy()
                vec2 = vec2 * 255
                r2 = vec2[0, ...]
                g2 = vec2[1, ...]
                b2 = vec2[2, ...]
                axs[1][0].plot(r2, color='r')
                axs[1][1].plot(g2, color='g')
                axs[1][2].plot(b2, color='b')

                vec3 = vec3.squeeze(0)
                # vec3 = vec3 * 0.5 + 0.5
                vec3 = vec3.cpu().detach().numpy()
                vec3 = vec3 * 255
                r3 = vec3[0, ...]
                g3 = vec3[1, ...]
                b3 = vec3[2, ...]
                axs[2][0].plot(r3, color='r')
                axs[2][1].plot(g3, color='g')
                axs[2][2].plot(b3, color='b')

                axs[0][0].set_xlim([0, 255])
                axs[0][0].set_ylim([0, 255])
                axs[0][1].set_xlim([0, 255])
                axs[0][1].set_ylim([0, 255])
                axs[0][2].set_xlim([0, 255])
                axs[0][2].set_ylim([0, 255])
                axs[1][0].set_xlim([0, 255])
                axs[1][0].set_ylim([0, 255])
                axs[1][1].set_xlim([0, 255])
                axs[1][1].set_ylim([0, 255])
                axs[1][2].set_xlim([0, 255])
                axs[1][2].set_ylim([0, 255])
                axs[2][0].set_xlim([0, 255])
                axs[2][0].set_ylim([0, 255])
                axs[2][1].set_xlim([0, 255])
                axs[2][1].set_ylim([0, 255])
                axs[2][2].set_xlim([0, 255])
                axs[2][2].set_ylim([0, 255])

                plt.tight_layout()
                plt.draw()
                plt.savefig(plot_path)

            sum_psnr += cal_PSNR(result_path)
    avg_psnr = sum_psnr / n_of_files
    print('Avg_PSNR: %.3f\t' % (avg_psnr))

    return avg_psnr


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            print(param)
            param.requires_grad = False
            print(param)
        dfs_freeze(child)


def train(config):
    sum_time = 0
    highest_psnr = 0
    highest_psnr_s = 0
    psnr_ep = 0

    if torch.cuda.is_available():
        cudnn.benchmark = True
    else:
        raise Exception("No GPU found, please run without --cuda")

    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)  # change allocation of current GPU
    Imgnet = Image_network().cuda()
    Imgnet.apply(weights_init)
    Imgnet = Imgnet.cuda()

    train_dataset = dataloader.input_loader(config.train_images_path)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                               num_workers=config.num_workers, pin_memory=True)

    loss_c = torch.nn.MSELoss().cuda()
    loss_e = Myloss.entropy_loss().cuda()
    cos = torch.nn.CosineSimilarity(dim=1)
    loss_t = Myloss.totalvariation_loss().cuda()

    optimizer_img = torch.optim.Adam(Imgnet.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    Imgnet.train()

    num_params = 0

    for param in Imgnet.parameters():
        num_params += param.numel()
    print('# of Imgnet params : %d' % num_params)

    cont_c = 0.5
    cont_e = 0.2
    cont_cs = 0.3

    lambda_c = 0
    lambda_e = 0
    lambda_cs = 0

    difficulty_c = 0
    difficulty_e = 0
    difficulty_cs = 0

    loss_col_0 = 0
    loss_ent_0 = 0
    loss_cos_0 = 0
    loss_0 = 0

    loss_col_2 = 0
    loss_ent_2 = 0
    loss_cos_2 = 0
    loss_2 = 0

    for epoch in range(config.num_epochs):
        st = time.time()
        print("epoch :", epoch + 1)
        sumLossCol = 0
        sumLossEnt = 0
        sumLossCos = 0
        sumLossTV = 0

        sumLoss = 0
        sumLoss_ = 0

        for iteration, (low, gt, hist) in enumerate(train_loader):

            low = low.cuda()
            gt = gt.cuda()
            hist = hist.cuda()

            img, tf, w, _ = Imgnet(low, hist)
            img = img.cuda()

            loss_img = loss_c(img, gt)
            loss_ent = loss_e(w)
            loss_col = torch.mean(1 - torch.abs(cos(gt, img)))
            loss_tv = loss_t(w)

            if epoch == 0:
                loss_f = (cont_c * loss_img + cont_e * loss_ent + loss_tv + cont_cs * loss_col)
            elif epoch == 1:
                loss_f = (lambda_c * loss_img) + (lambda_e * loss_ent) + (loss_tv) + (lambda_cs * loss_col)
            else:
                loss_f = (lambda_c * difficulty_c * loss_img) + (
                        lambda_e * difficulty_e * loss_ent) + (loss_tv) + (lambda_cs * difficulty_cs * loss_col)
            loss_ = loss_f - loss_tv

            optimizer_img.zero_grad()
            loss_f.backward()
            torch.nn.utils.clip_grad_norm_(Imgnet.parameters(), config.grad_clip_norm)
            optimizer_img.step()

            sumLossCol += loss_img.item()
            sumLossEnt += loss_ent.item()
            sumLossCos += loss_col.item()
            sumLossTV += loss_tv.item()

            sumLoss += loss_f.item()
            sumLoss_ += loss_.item()

            if iteration == (len(train_loader) - 1):
                print("Fus Loss:", loss_f.item())
                torch.save(Imgnet.state_dict(), config.snapshots_folder + "Img_tmp.pth")

                loss_col_0 = sumLossCol / len(train_loader)
                loss_ent_0 = sumLossEnt / len(train_loader)
                loss_cos_0 = sumLossCos / len(train_loader)
                loss_0 = sumLoss_ / len(train_loader)

                writer.add_scalar('color_loss', sumLossCol / len(train_loader), epoch + 1)
                writer.add_scalar('transformationFunction_loss', sumLossEnt / len(train_loader), epoch + 1)
                writer.add_scalar('cosineSimilarity_loss', sumLossCos / len(train_loader), epoch + 1)
                writer.add_scalar('totalVariation_loss', sumLossTV / len(train_loader), epoch + 1)
                writer.add_scalar('total_loss', sumLoss / len(train_loader), epoch + 1)

        if epoch == 0:
            loss_col_1 = loss_col_0
            loss_ent_1 = loss_ent_0
            loss_cos_1 = loss_cos_0
            loss_1 = loss_0
            # get loss weights
            lambda_c = cont_c * (loss_1 / loss_col_1)
            lambda_e = cont_e * (loss_1 / loss_ent_1)
            lambda_cs = cont_cs * (loss_1 / loss_cos_1)

            print()
            print('lambda_c\t' + str(lambda_c))
            print('lambda_e\t' + str(lambda_e))
            print('lambda_cs\t' + str(lambda_cs))

            print()
            # update previous losses
            loss_col_2 = loss_col_1
            loss_ent_2 = loss_ent_1
            loss_cos_2 = loss_cos_1

            loss_2 = loss_1


        else:
            loss_col_1 = loss_col_0
            loss_ent_1 = loss_ent_0
            loss_cos_1 = loss_cos_0
            loss_1 = loss_0
            # get loss weights
            lambda_c = cont_c * (loss_1 / loss_col_1)
            lambda_e = cont_e * (loss_1 / loss_ent_1)
            lambda_cs = cont_cs * (loss_1 / loss_cos_1)

            print()
            print('lambda_c\t' + str(lambda_c))
            print('lambda_e\t' + str(lambda_e))
            print('lambda_cs\t' + str(lambda_cs))
            print()
            # get difficulties
            difficulty_c = ((loss_col_1 / loss_col_2) / (loss_1 / loss_2)) ** config.beta
            difficulty_e = ((loss_ent_1 / loss_ent_2) / (loss_1 / loss_2)) ** config.beta
            difficulty_cs = ((loss_cos_1 / loss_cos_2) / (loss_1 / loss_2)) ** config.beta
            print('difficulty_c\t' + str(difficulty_c))
            print('difficulty_e\t' + str(difficulty_e))
            print('difficulty_cs\t' + str(difficulty_cs))
            print()
            # update previous losses
            loss_col_2 = loss_col_1
            loss_ent_2 = loss_ent_1
            loss_cos_2 = loss_cos_1
            loss_2 = loss_1

        psnr = eval("tmp", save_plot=False)
        if highest_psnr < psnr:
            highest_psnr = psnr
            psnr_ep = epoch + 1
            if not os.path.isdir("./data/best_score/best_psnr"):
                os.mkdir("./data/best_score/best_psnr")
            copy_tree("./data/train_check/test", "./data/best_score/best_psnr")
            shutil.copy("./models/Img_tmp.pth", "./models/Img_final.pth")

        writer.add_scalar('PSNR', psnr, epoch + 1)
        et = time.time() - st
        print('%d epoch: %.3f' % (epoch + 1, et))
        sum_time += et
        rTime = (sum_time / (epoch + 1)) * (config.num_epochs - (epoch + 1))
        print("Estimated time remaining :%d hour %d min %d sec" % (
            rTime / 3600, (rTime % 3600) / 60, (rTime % 3600) % 60))

    print('Hightest PSNR: ' + str(highest_psnr) + '\tSSIM: ' + str(highest_psnr_s) + '\t(Epoch' + str(psnr_ep) + ')')
    _ = eval("final", save_plot=True)

    f = open('./data/best_score/best_scores.txt', 'w')
    sys.stdout = f
    print('Hightest PSNR: ' + str(highest_psnr) + '\tSSIM: ' + str(highest_psnr_s) + '\t(Epoch' + str(psnr_ep) + ')')
    sys.stdout = sys.__stdout__
    f.close()


if __name__ == "__main__":
    start_time = time.time()
    # using_cuda()
    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--train_images_path', type=str, default="./data/train_data/")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.00005)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--beta', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, default="models/")
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--gpus', default=1, type=int, help='number of gpu')

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)

    train(config)
    writer.close()
    total_time = time.time() - start_time
    print("total = %dhour %dmin %dsec" % (total_time / 3600, (total_time % 3600) / 60, (total_time % 3600) % 60))
