import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import os
import sys
import shutil
import math
import random
import heapq 
import time
import copy
import itertools  
from typing import Dict, List
from PIL import Image
from io import StringIO,BytesIO 
from scipy.spatial.distance import pdist
from scipy.signal import butter, lfilter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize,normalize
from sklearn.metrics import confusion_matrix,roc_curve,accuracy_score,auc,roc_auc_score 
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.ndimage import zoom
from functools import reduce
from scipy.io import loadmat
from skimage.measure import block_reduce
from collections import Counter
from scipy.sparse import coo_matrix,hstack, vstack
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.ops as ops

'''
#Code: https://github.com/PyRetri/PyRetri
       https://github.com/YahooArchive/crow
#Paper: ECCV2017《Cross-dimensional Weighting for Aggregated Deep Convolutional Features》
'''
class CroW():
    """
    Cross-dimensional Weighting for Aggregated Deep Convolutional Features.
    c.f. https://arxiv.org/pdf/1512.04065.pdf
    Args:
        spatial_a (float): hyper-parameter for calculating spatial weight.
        spatial_b (float): hyper-parameter for calculating spatial weight.
    """

    def __init__(self, spatial_a=2.0, spatial_b=2.0):
       
        self.first_show = True
        self.spatial_a = spatial_a
        self.spatial_b = spatial_b

    def __call__(self, fea:torch.tensor) -> torch.tensor:
        final_fea = None
        if fea.ndimension() == 4:
            spatial_weight = fea.sum(dim=1, keepdim=True)
            z = (spatial_weight ** self.spatial_a).sum(dim=(2, 3), keepdim=True)
            z = z ** (1.0 / self.spatial_a)
            spatial_weight = (spatial_weight / z) ** (1.0 / self.spatial_b)

            c, w, h = fea.shape[1:]
            nonzeros = (fea!=0).float().sum(dim=(2, 3)) / 1.0 / (w * h) + 1e-6
            channel_weight = torch.log(nonzeros.sum(dim=1, keepdim=True) / nonzeros)

            fea = fea * spatial_weight
            fea = fea.sum(dim=(2, 3))
            fea = fea * channel_weight
            
            final_fea = fea

        else:# In case of fc feature.
            assert fea.ndimension() == 2
            if self.first_show:
                print("[Crow Aggregator]: find 2-dimension feature map, skip aggregation")
                self.first_show = False
            final_fea = fea
            
        return final_fea

if __name__ == "__main__":
    #for debug   
    x = torch.rand(2, 3, 256, 256)#.to(torch.device('cuda:%d'%7))
    model = DSH(num_binary=64)#.to(torch.device('cuda:%d'%7))
    out = model(x)
    print(out.size())