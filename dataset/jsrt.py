import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import pandas as pd
import numpy as np
import time
import random
from sklearn.model_selection import train_test_split
import sys
import torch.nn.functional as F
import scipy
import SimpleITK as sitk
import pydicom
from scipy import ndimage as ndi
from PIL import Image
import PIL.ImageOps 
import cv2
from scipy.io import loadmat
"""
Dataset: JSRT CXR
https://www.kaggle.com/raddar/nodules-in-chest-xrays-jsrt
"""

class DatasetGenerator(Dataset):
    def __init__(self, path_to_img_dir, path_to_dataset_file):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        image_names = []
        mask_names = []
        labels = []
        for file_path in path_to_dataset_file:
            with open(file_path, "r") as f:
                for line in f: 
                    items = line.strip().split(',') 
                    image_name = os.path.join(path_to_img_dir, 'images', items[0])
                    if items[1] == 'l':
                        mask_name = os.path.join(path_to_img_dir,'masks/left_lung/', os.path.splitext(items[0])[0]+'.gif')
                    else: #right

                        mask_name = os.path.join(path_to_img_dir,'masks/right_lung/', os.path.splitext(items[0])[0]+'.gif')
                    if os.path.isfile(image_name) == True and os.path.isfile(mask_name) == True:
                        label = int(eval(items[2])) #eval for 
                        image_names.append(image_name)    
                        mask_names.append(mask_name)
                        labels.append([label])

        self.image_names = image_names
        self.mask_names = mask_names
        self.labels = labels
        self.transform_seq_image = transforms.Compose([
            transforms.Resize((256,256)),#256
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            ])
        self.transform_seq_mask = transforms.Compose([
            transforms.Resize((256,256))
            ])

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        try:
            image_name = self.image_names[index]
            image = Image.open(image_name).convert('RGB')
            image = self.transform_seq_image(image)
            
            mask_name = self.mask_names[index]
            mask = Image.open(mask_name)
            mask = torch.LongTensor(np.array(self.transform_seq_mask(mask)))
            #turn to 0-0, else=2, 255=1
            mask = torch.where(mask==255, torch.full_like(mask, -1), mask) 
            mask = torch.where(mask>0, torch.full_like(mask, 2), mask)
            mask = torch.where(mask==-1, torch.full_like(mask, 1), mask)

            label = torch.FloatTensor(self.labels[index])
            #label = torch.LongTensor(self.labels[index])
        except Exception as e:
            print("Unable to read file. %s" % e)
        
        return image, mask, label

    def __len__(self):
        return len(self.labels)

#config 
PATH_TO_IMAGES_DIR = '/data/fjsdata/JSRT-CXR/'
PATH_TO_TRAIN_FILE = '/data/fjsdata/JSRT-CXR/jsrt_train.txt'
PATH_TO_TEST_FILE = '/data/fjsdata/JSRT-CXR/jsrt_test.txt'

def get_train_dataloader(batch_size, shuffle, num_workers):
    dataset_train = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR,path_to_dataset_file=[PATH_TO_TRAIN_FILE])
    #sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train) #for multi cpu and multi gpu
    #data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, sampler = sampler_train, 
                                   #shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, pin_memory=True)#drop_last=True
    return data_loader_train

def get_test_dataloader(batch_size, shuffle, num_workers):
    dataset_test = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR,path_to_dataset_file=[PATH_TO_TEST_FILE])
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_test


def splitDataset(dataset_path):
    #deal with true positive samples
    datas = pd.read_csv(dataset_path, sep=',')
    print("\r CXR Columns: {}".format(datas.columns))
    datas = datas[['study_id', 'state', 'position']]
    datas = datas.drop(datas[datas['state']=='non-nodule'].index)
    print("\r Dataset shape: {}".format(datas.shape)) 
    datas['state'] = datas['state'].apply(lambda x: 0 if x == 'benign' else 1) #benign=0 or malignant=1
    datas['position'] = datas['position'].apply(lambda x: x[0:1]) #left lung or right lung
    
    images = datas[['study_id', 'position']]
    labels = datas[['state']]
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.10, random_state=11)
    print("\r trainset shape: {}".format(X_train.shape)) 
    print("\r trainset distribution: {}".format(y_train['state'].value_counts()))
    print("\r testset shape: {}".format(X_test.shape)) 
    print("\r trainset distribution: {}".format(y_test['state'].value_counts()))
    trainset = pd.concat([X_train, y_train], axis=1).to_csv('/data/fjsdata/JSRT-CXR/jsrt_train.txt', index=False, header=False, sep=',')
    testset = pd.concat([X_test, y_test], axis=1).to_csv('/data/fjsdata/JSRT-CXR/jsrt_test.txt', index=False, header=False, sep=',')


if __name__ == "__main__":
  
    splitDataset('/data/fjsdata/JSRT-CXR/fjs_jsrt.csv')
    
    #for debug   
    data_loader_train = get_train_dataloader(batch_size=10, shuffle=True, num_workers=0)
    for batch_idx, (image, mask, label) in enumerate(data_loader_train):
        print(batch_idx)
        print(image.shape)
        print(mask.shape)
        print(label.shape)
        break
    