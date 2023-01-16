import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import torchvision

import numpy as np
from PIL import Image


class perceptual_loss(nn.Module):
    def __init__(self, requires_grad=False):
        super(perceptual_loss, self).__init__()

        self.maeloss = torch.nn.L1Loss()
        vgg = vgg16(pretrained=True).cuda()

        vgg_pretrained_features = vgg.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 4):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 6):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = True

    def forward(self, X, Y):
        # print(X.shape)
        xx = self.slice1(X)
        fx2 = xx
        xx = self.slice2(xx)
        fx4 = xx
        xx = self.slice3(xx)
        fx6 = xx

        yy = self.slice1(Y)
        fy2 = yy
        yy = self.slice2(yy)
        fy4 = yy
        yy = self.slice3(yy)
        fy6 = yy

        loss_p = self.maeloss(fx2, fy2) + self.maeloss(fx4, fy4) + self.maeloss(fx6, fy6)

        return loss_p


class monotonous_loss(nn.Module):
    def __init__(self, requires_grad=False):
        super(monotonous_loss, self).__init__()

    def forward(self, v):
        b = v[0]
        batch_size = b.size()[0]
        # x = x * 0.5 + 0.5
        # x = torch.round(255 * x)
        loss_m = 0

        for n in range(0, len(v)):
            x = v[n]
            g_sum = torch.zeros(batch_size, 3).cuda()
            for i in range(0, 255):
                g = x[:, :, i + 1] - x[:, :, i]
                h = g.clone()
                g[g < 0] = 1.
                g[h >= 0] = 0.
                g_sum += g
            g_sum = g_sum/255
            loss_m += torch.mean(g_sum)

        # print(loss_m/3)

        return loss_m


class attention_loss(nn.Module):
    def __init__(self):
        super(attention_loss, self).__init__()
        self.mseloss = nn.L1Loss().cuda()

    def forward(self, att, att_gt):
        # att_gt = att_gt.unsqueeze(1)
        # att = att * 0.5 + 0.5
        att_gt = F.interpolate(att_gt, 256, mode='bilinear', align_corners=True)
        # print(att_gt)

        return self.mseloss(att, att_gt)

class transfunction_loss(nn.Module):
    def __init__(self):
        super(transfunction_loss, self).__init__()
        self.maeloss = nn.L1Loss().cuda()

    def forward(self, tf, htf, gt):

        return 0.6*self.maeloss(tf, gt) + 0.4*self.maeloss(htf, gt)


class entropy_loss(nn.Module):
    def __init__(self):
        super(entropy_loss, self).__init__()

    def forward(self, w):
        # print(w[0].shape)
        b = w[0]
        batch_size = b.size()[0]
        e_sum = torch.zeros(batch_size, b.size()[1], b.size()[2], b.size()[3]).cuda()
        for n in range(0,len(w)):
            # print(w[n])
            ent = -w[n] * torch.log2(w[n])
            e_sum += ent
        e_sum = e_sum
        loss_e = torch.mean(e_sum)
        # print(loss_e)

        return loss_e

class totalvariation_loss(nn.Module):
    def __init__(self, TVLoss_weight=1e-4):
        super(totalvariation_loss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,w0):
        w1, w2, w3 = w0
        w = torch.cat((w1,w2,w3),dim=1)
        loss_tv = 0
        batch_size = w.size()[0]
        h_x = w.size()[2]
        w_x = w.size()[3]
        count_h = (w.size()[2] - 1) * w.size()[3]
        count_w = w.size()[2] * (w.size()[3] - 1)

        h_tv = torch.pow((w[:,:,1:,:]-w[:,:,:h_x-1,:]),2).sum() / count_h
        w_tv = torch.pow((w[:,:,:,1:]-w[:,:,:,:w_x-1]),2).sum() / count_w
        loss_tv = self.TVLoss_weight * (h_tv + w_tv) / batch_size

        # for x in w:
        #     h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        #     w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        #     loss_tv += self.TVLoss_weight * (h_tv + w_tv)

        return loss_tv

