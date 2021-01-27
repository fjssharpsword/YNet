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
Dataset: Fundus Origa650
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
                    iname=os.path.splitext(items[0])[0].strip()[1:]
                    image_name = os.path.join(path_to_img_dir, 'images', iname+'.jpg')
                    mask_name = os.path.join(path_to_img_dir,'mask', iname+'.mat')
                    if os.path.isfile(image_name) == True and os.path.isfile(mask_name) == True:
                        label = int(eval(items[1])) #eval for 
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
            mask = Image.fromarray(loadmat(mask_name)['mask'])
            mask = torch.LongTensor(np.array(self.transform_seq_mask(mask)))

            label = torch.FloatTensor(self.labels[index])
        except Exception as e:
            print("Unable to read file. %s" % e)
        
        return image, mask, label

    def __len__(self):
        return len(self.labels)

#config 


PATH_TO_IMAGES_DIR = '/data/fjsdata/MCBIR-Ins/origa650/'
PATH_TO_TRAIN_FILE = '/data/fjsdata/MCBIR-Ins/origa650/trainset.csv'
PATH_TO_TEST_FILE = '/data/fjsdata/MCBIR-Ins/origa650/testset.csv'

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



if __name__ == "__main__":
  
    #for debug   
    data_loader_train = get_train_dataloader(batch_size=10, shuffle=True, num_workers=0)
    for batch_idx, (image, mask, label) in enumerate(data_loader_train):
        print(batch_idx)
        print(image.shape)
        print(mask.shape)
        print(label.shape)