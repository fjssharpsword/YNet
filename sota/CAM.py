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
#Code: https://github.com/imatge-upc/retrieval-2017-cam
#Paper: BMVC2017《Class-Weighted Convolutional Features for Image Retrieval》
'''
# Extract region of interest from CAMs
def extract_ROI(heatmap, threshold):
    th = threshold * np.max(heatmap)
    heatmap = heatmap > th
    # Find the largest connected component

    contours, hierarchy = cv2.findContours(heatmap.astype('uint8'), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    areas = [cv2.contourArea(ctr) for ctr in contours]

    max_contour = contours[areas.index(max(areas))]

    x, y, w, h = cv2.boundingRect(max_contour)
    if w == 0:
        w = heatmap.shape[1]
    if h == 0:
        h = heatmap.shape[0]
    return x, y, w, h

def compute_crow_channel_weight(X):
    """
    Given a tensor of features, compute channel weights as the
    log of inverse channel sparsity.
    :param ndarray X:
        3d tensor of activations with dimensions (channels, height, width)
    :returns ndarray:
        a channel weight vector
    """
    K, w, h = X.shape
    area = float(w * h)
    nonzeros = np.zeros(K, dtype=np.float32)
    for i, x in enumerate(X):
        nonzeros[i] = np.count_nonzero(x) / area

    nzsum = nonzeros.sum()
    for i, d in enumerate(nonzeros):
        nonzeros[i] = np.log(nzsum / d) if d > 0. else 0.

    return nonzeros

def compute_pca(descriptors, pca_dim=512, whiten=True):
    print (descriptors.shape)
    t1 = time.time()
    print( 'Computing PCA with dimension reduction to: ', pca_dim)
    sys.stdout.flush()
    pca = PCA(n_components=pca_dim, whiten=whiten)
    pca.fit(descriptors)
    print (pca.components_.shape)
    print ('PCA finished!')
    print ('Time elapsed computing PCA: ', time.time() - t1)
    return pca


def sum_pooling(features):
    num_samples = features.shape[0]
    num_features = features.shape[1]
    sys.stdout.flush()
    descriptors = np.zeros((num_samples, num_features), dtype=np.float32)
    for i in range(0, num_samples):
        #print 'Image: ', i
        #sys.stdout.flush()
        for f in range(0, num_features):
            descriptors[i, f] = features[i, f].sum()
    descriptors /= np.linalg.norm(descriptors, axis=1)[:, None]
    return descriptors

def weighted_cam_pooling(features, cams, max_pool=False, channel_weights=True):
    '''
    :param features: Feature Maps
    :param cams: Class Activation Maps
    :param max_pool: Perform also Max pooling
    :param channel_weights: Channel Weighting as in Crow
    :return: A descriptor for each CAM.
    '''
    t = time.time()
    num_samples = features.shape[0]
    num_features = features.shape[1]
    num_classes = cams.shape[1]

    wp_regions = np.zeros((num_features, num_classes), dtype=np.float32)
    wsp_descriptors_reg = np.zeros((num_samples * num_classes, num_features), dtype=np.float32)
    wmp_descriptors_reg = np.zeros((num_samples * num_classes, num_features), dtype=np.float32)

    if max_pool:
        mp_regions = np.zeros((num_features, num_classes), dtype=np.float32)

    for i in range(0, num_samples):
        #CROW
        if channel_weights:
            C = np.array(compute_crow_channel_weight(features[i]))

        for f in range(0, num_features):
            for k in range(0, num_classes):
                # For each region compute avg weighted sum of activations and l2 normalize
                if max_pool:
                        mp_regions[f, k] = np.amax(np.multiply(features[i, f], cams[i, k]))
                wp_regions[f, k] = np.multiply(features[i, f], cams[i, k]).sum()

        if channel_weights:
            wp_regions = wp_regions * C[:, None]
        wp_regions /= np.linalg.norm(wp_regions, axis=0)

        if max_pool:
            if channel_weights:
                mp_regions = mp_regions * C[:, None]
            mp_regions /= np.linalg.norm(mp_regions, axis=0)

        wsp_descriptors_reg[num_classes*i:num_classes*(i+1)] = np.transpose(wp_regions)

        if max_pool:
            wmp_descriptors_reg[num_classes*i:num_classes*(i+1)] = np.transpose(mp_regions)

    #print 'Time elapsed computing image representations for the batch: ', time.time() - t

    if max_pool:
        return wsp_descriptors_reg, wmp_descriptors_reg
    else:
        return wsp_descriptors_reg
    
# General Descriptor Aggregation : PCA + Aggregation
def descriptor_aggregation(descriptors_cams, num_images, num_classes, pca=None):

    num_classes_ori = int(descriptors_cams.shape[0] / num_images)
    descriptors = np.zeros((num_images, descriptors_cams.shape[1]), dtype=np.float32)

    if pca is not None:
        # Sometimes we may have errors during re-ranking due to bounding box generation on places where CAM=0
        try:
            descriptors_pca = pca.transform(descriptors_cams)
        except:
            print ('---------------------------->Exception')
            desc_err = np.zeros((descriptors_cams.shape[0], descriptors_cams.shape[1]), dtype=np.float32)
            for j in range(0, descriptors_cams.shape[0]):
                try:
                    desc_err[j] = pca.transform(descriptors_cams[j])
                except:
                    print ('---------------------------->Exception')
                    print (j)
                    desc_err[j] = desc_err[j-1]
            descriptors_pca = desc_err

        descriptors = np.zeros((num_images, descriptors_pca.shape[1]), dtype=np.float32)
        #print descriptors_pca.shape

    index = 0
    for i in range(0, num_images):
        index = num_classes_ori + index
        if i == 0:
            index = 0
        if pca is not None:
            for k in range(index, index+num_classes):
                descriptors_pca[k] /= np.linalg.norm(descriptors_pca[k])
                descriptors[i] += descriptors_pca[k]

            descriptors[i] /= np.linalg.norm(descriptors[i])
        else:
            for k in range(index, index+num_classes):
                descriptors[i] += descriptors_cams[k]
            descriptors[i] /= np.linalg.norm(descriptors[i])

    return descriptors

if __name__ == "__main__":
    #for debug   
    x = torch.rand(2, 3, 256, 256)#.to(torch.device('cuda:%d'%7))
    model = DSH(num_binary=64)#.to(torch.device('cuda:%d'%7))
    out = model(x)
    print(out.size())