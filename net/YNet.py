# encoding: utf-8
"""
YNet for medical instance retrieval
Author: Jason.Fang
Update time: 28/12/2020
"""
import sys
import re
import numpy as np
import random
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from skimage.measure import label as skmlabel
import cv2
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import torchvision

class L2Normalization(nn.Module):
    def __init__(self):
        """
        In the constructor we construct three nn.Linear instances that we will use
        in the forward pass.
        """
        super(L2Normalization, self).__init__()
        self.eps = 1e-8
        
    def forward(self, x):
        if x.is_cuda:
            caped_eps = Variable(torch.Tensor([self.eps])).cuda(torch.cuda.device_of(x).idx)
        else:
            caped_eps = Variable(torch.Tensor([self.eps]))
        x = torch.div(x.transpose(0,1),x.max(1)[0]).transpose(0,1) # max_normed
        norm = torch.norm(x,2,1) + caped_eps.expand(x.size()[0])
        y = torch.div(x.transpose(0,1),norm).transpose(0,1)
        return y

class RMAC(nn.Module):
    """
    Regional Maximum activation of convolutions (R-MAC).
    c.f. https://arxiv.org/pdf/1511.05879.pdf
    Args:
        level_n (int): number of levels for selecting regions.
    """
    def __init__(self,level_n:int):
        super(RMAC, self).__init__()
        self.first_show = True
        self.cached_regions = dict()
        self.level_n = level_n

    def _get_regions(self, h: int, w: int) -> list:
        #Divide the image into several regions.
        #Args:
        #    h (int): height for dividing regions.
        #    w (int): width for dividing regions.
        #Returns:
        #    regions (List): a list of region positions.
   
        if (h, w) in self.cached_regions:
            return self.cached_regions[(h, w)]

        m = 1
        n_h, n_w = 1, 1
        regions = list()
        if h != w:
            min_edge = min(h, w)
            left_space = max(h, w) - min(h, w)
            iou_target = 0.4
            iou_best = 1.0
            while True:
                iou_tmp = (min_edge ** 2 - min_edge * (left_space // m)) / (min_edge ** 2)

                # small m maybe result in non-overlap
                if iou_tmp <= 0:
                    m += 1
                    continue

                if abs(iou_tmp - iou_target) <= iou_best:
                    iou_best = abs(iou_tmp - iou_target)
                    m += 1
                else:
                    break
            if h < w:
                n_w = m
            else:
                n_h = m

        for i in range(self.level_n):
            region_width = int(2 * 1.0 / (i + 2) * min(h, w))
            step_size_h = (h - region_width) // n_h
            step_size_w = (w - region_width) // n_w

            for x in range(n_h):
                for y in range(n_w):
                    st_x = step_size_h * x
                    ed_x = st_x + region_width - 1
                    assert ed_x < h
                    st_y = step_size_w * y
                    ed_y = st_y + region_width - 1
                    assert ed_y < w
                    regions.append((st_x, st_y, ed_x, ed_y))

            n_h += 1
            n_w += 1

        self.cached_regions[(h, w)] = regions
        return regions

    def forward(self, fea:torch.tensor) -> torch.tensor:
        final_fea = None
        if fea.ndimension() == 4:
            h, w = fea.shape[2:]       
            regions = self._get_regions(h, w)
            for _, r in enumerate(regions):
                st_x, st_y, ed_x, ed_y = r
                region_fea = (fea[:, :, st_x: ed_x, st_y: ed_y].max(dim=3)[0]).max(dim=2)[0]#max-pooling
                region_fea = region_fea / torch.norm(region_fea, dim=1, keepdim=True)#PCA-whitening
                if final_fea is None:
                    final_fea = region_fea
                else:
                    final_fea = final_fea + region_fea
        else:# In case of fc feature.
            assert fea.ndimension() == 2
            if self.first_show:
                print("[RMAC Aggregator]: find 2-dimension feature map, skip aggregation")
                self.first_show = False
            final_fea = fea
        return final_fea

class CircleLoss(nn.Module):
    #sampling all pospair and negpair
    def __init__(self, scale=1, margin=0.25, similarity='cos', **kwargs):
        super(CircleLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.similarity = similarity

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"

        mask = torch.matmul(labels, torch.t(labels))
        pos_mask = mask.triu(diagonal=1)
        neg_mask = (mask - 1).abs_().triu(diagonal=1)
        if self.similarity == 'dot':
            sim_mat = torch.matmul(feats, torch.t(feats))
        elif self.similarity == 'cos':
            feats = F.normalize(feats)
            sim_mat = feats.mm(feats.t())
        else:
            raise ValueError('This similarity is not implemented.')

        pos_pair_ = sim_mat[pos_mask == 1]
        neg_pair_ = sim_mat[neg_mask == 1]
        alpha_p = torch.relu(-pos_pair_ + 1 + self.margin)
        alpha_n = torch.relu(neg_pair_ + self.margin)
        margin_p = 1 - self.margin
        margin_n = self.margin
        loss_p = torch.sum(torch.exp(-self.scale * alpha_p * (pos_pair_ - margin_p)))
        loss_n = torch.sum(torch.exp(self.scale * alpha_n * (neg_pair_ - margin_n)))
        loss = torch.log(1 + loss_p * loss_n)
        return loss

class YNetLoss(nn.Module):
    def __init__(self, lr=0.001):
        super(YNetLoss, self).__init__()
        self.alpha = nn.Parameter(torch.FloatTensor([0.5]))
        #self.beta = nn.Parameter(torch.FloatTensor([1.0]))
        #self.l1_reg = nn.Parameter(F.normalize(torch.tensor([torch.abs(self.alpha), torch.abs(self.beta)]), p=1, dim=0))
        self.lr = lr #weight_decay
        
    def forward(self,clloss, celoss): 
        #l2_reg = self.lr* (torch.norm(torch.abs(self.alpha), p=2) + torch.norm(torch.abs(1-self.alpha), p=2))
        #l2_reg = self.lr* (torch.norm(torch.abs(self.alpha), p=2)
        loss = torch.abs(self.alpha)*clloss + torch.abs(1-self.alpha)*celoss #+ l2_reg
        return loss

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class YNet(nn.Module):
    def __init__(self, block=Bottleneck, num_blocks=[2,2,2,2], n_classes=2, n_masks=3, code_size=64):
        super(YNet, self).__init__()
        # Bottom-up layers，classifcation loss
        self.in_planes = 8  #3 D->64 channels
        self.conv1 = nn.Conv2d(3, 8, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(8)

        self.layer2 = self._make_layer(block, 8, num_blocks[0], stride=1)
        self.layer3 = self._make_layer(block, 16, num_blocks[1], stride=2)
        self.layer4 = self._make_layer(block, 32, num_blocks[2], stride=2)
        self.layer5 = self._make_layer(block, 64, num_blocks[3], stride=2)
        self.conv6 = nn.Conv2d(256, 32, kernel_size=3, stride=2, padding=1)
        
        #self.fc1 = nn.Linear(32*4*4, code_size)#code_size:length of hash code
        self.fc1 = nn.Linear(256*8*8, code_size)
        self.fc2 = nn.Linear(code_size, n_classes) #num_classes:number of classes
        
        # Top-down layer, segmentation loss
        self.toplayer = nn.Conv2d(256, 32, kernel_size=1, stride=1, padding=0)  # Reduce channels
        
        self.latlayer1 = nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0)# Lateral layers
        self.latlayer2 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)
        
        self.upsample = nn.Upsample((256,256), mode='bilinear',align_corners=True)
        self.conv7 = nn.Conv2d(32, n_masks, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.ReLU() #nn.BatchNorm2d(n_masks)  #mask 0,1,2
        
        # Hash layer, ranking loss
        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.r_mac_pool = RMAC(level_n=3) 
        self.l2norm = L2Normalization()
        self.fc3 = nn.Linear(512, n_classes) #num_classes:number of classes

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        # Bottom-up, classifcation loss
        h1 = F.relu(self.bn1(self.conv1(x)))#(3,256,256)->(8,128,128)
        h1 = F.max_pool2d(h1, kernel_size=3, stride=2, padding=1)#(8,128,128)->(8,64,64)
        
        h2 = self.layer2(h1)#(8,64,64)->(32,64,64)
        h3 = self.layer3(h2)#(32,64,64)->(64,32,32)
        h4 = self.layer4(h3)#(64,32,32)->(128,16,16)
        h5 = self.layer5(h4)#(128,16,16)->(256,8,8)
        
        #h6 = self.conv6(h5)#(256,8,8)->(32,4,4) 
        h6 = h5.view(h5.size(0), -1)#conv->linear
        h_feat = self.fc1(h6)
        h_cls = self.fc2(h_feat)
        
        # Top-down, segmentation loss
        s5 = self.toplayer(h5)#(256,8,8)->(32,8,8)
        s4 = self._upsample_add(s5, self.latlayer1(h4))#{(32,8,8),(32, 16, 16)}->(32, 16, 16)
        s3 = self._upsample_add(s4, self.latlayer2(h3))#{(32, 16, 16),(32, 32, 32)}->(32, 32, 32)
        s2 = self._upsample_add(s3, h2) #{(32, 32, 32),(32, 64, 64)}->(32, 64, 64)
        
        s1 = self.upsample(s2)#(32, 64, 64)->(32, 256, 256)
        s_mask = self.bn2(self.conv7(s1))#(32, 256, 256)->(n_masks, 256, 256)
        
        #Hash, ranking loss
        c5 = self.conv8(h5)#(256,8,8)->(512,8,8)
        
        c4 = self.r_mac_pool(c5) 
        
        c_feat = self.l2norm(c4) #512
        c_cls = self.fc3(c_feat)

        return h_feat, h_cls, s_mask, c_feat, c_cls

if __name__ == "__main__":
    #for debug   
    x = torch.rand(2, 3, 256, 256)#.to(torch.device('cuda:%d'%7))
    model = YNet(n_classes=2, n_masks=3, code_size=64)#.to(torch.device('cuda:%d'%7))
    h_feat, h_cls, s_mask, c_feat, c_cls = model(x)
    print(s_mask.size())