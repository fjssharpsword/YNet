# encoding: utf-8
"""
Training implementation
Author: Jason.Fang
Update time: 16/12/2020
"""
import re
import sys
import os
import cv2
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from skimage.measure import label
from sklearn.metrics.pairwise import cosine_similarity
import heapq
from sklearn.metrics import accuracy_score, confusion_matrix
#self-defined
from dataset.origa import get_train_dataloader, get_test_dataloader
#from dataset.jsrt import get_train_dataloader, get_test_dataloader
from util.Evaluation import compute_AUCs, compute_fusion
from net.YNet import YNet, YNetLoss, CircleLoss

#command parameters
parser = argparse.ArgumentParser(description='Y-Net')
parser.add_argument('--model', type=str, default='YNet', help='YNet')
args = parser.parse_args()

#config
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
MAX_EPOCHS = 20#500
BATCH_SIZE = 36

def Train():
    print('********************load data********************')
    dataloader_train = get_train_dataloader(batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    if args.model == 'YNet':
        model = model = YNet(n_classes=2, n_masks=3, code_size=64).cuda()
        #model = nn.DataParallel(model).cuda()  # make model available multi GPU cores training
        optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        lr_scheduler_model = lr_scheduler.StepLR(optimizer , step_size = 10, gamma = 1)
    else: 
        print('No required model')
        return #over
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    ce_loss  = nn.CrossEntropyLoss() #for segmentation
    cl_loss = CircleLoss() #for classification
    ynet_loss = YNetLoss().cuda() #for unified loss
    print('********************load model succeed!********************')

    print('********************begin training!********************')
    best_loss = float('inf')
    for epoch in range(MAX_EPOCHS):
        since = time.time()
        print('Epoch {}/{}'.format(epoch+1 , MAX_EPOCHS))
        print('-' * 10)
        model.train()  #set model to training mode
        train_loss = []
        with torch.autograd.enable_grad():
            for batch_idx, (image, mask, label) in enumerate(dataloader_train):
                optimizer.zero_grad()
                #forward
                var_image = torch.autograd.Variable(image).cuda()
                var_mask = torch.autograd.Variable(mask).cuda()
                var_label = torch.autograd.Variable(label).cuda()
                h_feat, h_cls, s_mask, c_feat, c_cls = model(var_image)
                #backward
                #label = torch.LongTensor(label.numpy()) #float to long
                #label = torch.autograd.Variable(label).cuda() #cpu to gpu
                #loss_tensor =  ce_loss(c_cls, label.squeeze())  #

                cls_loss = cl_loss(c_feat, var_label)
                mask_loss = ce_loss(s_mask, var_mask)
                loss_tensor = ynet_loss(cls_loss, mask_loss)
                
                loss_tensor.backward() 
                optimizer.step()
                #print([x.grad for x in optimizer.param_groups[0]['params']])
                sys.stdout.write('\r Epoch: {} / Step: {} : train loss = {}'.format(epoch+1, batch_idx+1, float('%0.6f'%loss_tensor.item()) ))
                sys.stdout.flush()       
                train_loss.append(loss_tensor.item()) 
        lr_scheduler_model.step()  #about lr and gamma
        print("\r Eopch: %5d train loss = %.6f" % (epoch + 1, np.mean(train_loss))) 

        if best_loss > np.mean(train_loss):
            best_loss = np.mean(train_loss)
            CKPT_PATH = '/data/pycode/YNet/model/best_model.pkl'
            torch.save(model.state_dict(), CKPT_PATH)
            #torch.save(model.module.state_dict(), CKPT_PATH) #Saving torch.nn.DataParallel Models
            print(' Epoch: {} model has been already save!'.format(epoch+1))

        time_elapsed = time.time() - since
        print('Training epoch: {} completed in {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60 , time_elapsed % 60))

def Test():
    print('********************load data********************')
    dataloader_train = get_train_dataloader(batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    dataloader_test = get_test_dataloader(batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    if args.model == 'YNet':
        model = model = YNet(n_classes=2, n_masks=3, code_size=64).cuda()
        CKPT_PATH = '/data/pycode/YNet/model/best_model.pkl'
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> loaded Image model checkpoint: "+CKPT_PATH)
    else: 
        print('No required model')
        return #over
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    model.eval() #turn to test mode
    print('******************** load model succeed!********************')

    print('******* begin indexing!*********')
    tr_label = torch.FloatTensor().cuda()
    #tr_mask = torch.LongTensor().cuda()
    tr_hash = torch.FloatTensor().cuda()
    with torch.autograd.no_grad():
        for batch_idx, (image, mask, label) in enumerate(dataloader_train):
            tr_label = torch.cat((tr_label, label.cuda()), 0)
            #tr_mask = torch.cat((tr_mask, mask.cuda()), 0)
            var_image = torch.autograd.Variable(image).cuda()
            h_feat, _, _, _, _ = model(var_image)
            tr_hash = torch.cat((tr_hash, h_feat.data), 0)
            sys.stdout.write('\r train indexing process: = {}'.format(batch_idx+1))
            sys.stdout.flush()
    
    te_label = torch.FloatTensor().cuda()
    te_mask = torch.LongTensor().cuda()
    te_hash = torch.FloatTensor().cuda()
    te_mask_pd = torch.LongTensor().cuda()
    te_label_pd = torch.LongTensor().cuda()
    with torch.autograd.no_grad():
        for batch_idx, (image, mask, label) in enumerate(dataloader_test):
            te_label = torch.cat((te_label, label.cuda()), 0)
            te_mask = torch.cat((te_mask, mask.cuda()), 0)
            var_image = torch.autograd.Variable(image).cuda()
            h_feat, _, s_mask, _, c_cls = model(var_image)
            te_hash = torch.cat((te_hash, h_feat.data), 0)
            s_mask = F.log_softmax(s_mask,dim=1) 
            s_mask = s_mask.max(1,keepdim=True)[1]
            te_mask_pd = torch.cat((te_mask_pd, s_mask.data), 0)
            c_cls = F.log_softmax(c_cls,dim=1) 
            c_cls = c_cls.max(1,keepdim=True)[1]
            te_label_pd = torch.cat((te_label_pd, c_cls.data), 0)
            sys.stdout.write('\r test indexing process: = {}'.format(batch_idx+1))
            sys.stdout.flush()

    print('******* begin evaluating!*********')
    #retrieval performance
    sim_mat = cosine_similarity(te_hash.cpu().numpy(), tr_hash.cpu().numpy())
    te_label = te_label.cpu().numpy().tolist()
    tr_label = tr_label.cpu().numpy().tolist()
    for topk in [5,10,20,50]:
        mAPs = [] #mean average precision
        for i in range(sim_mat.shape[0]):
            #idxs = heapq.nlargest(topk, sim_mat[i,:])
            idxs, vals = zip(*heapq.nlargest(topk, enumerate(sim_mat[i,:].tolist()), key=lambda x:x[1]))
            num_pos = 0
            rank_pos = 0
            mAP = []
            for j in idxs:
                rank_pos = rank_pos + 1
                if tr_label[j]==te_label[i]:  #hit
                    num_pos = num_pos +1
                    mAP.append(num_pos/rank_pos) 
            if np.sum(mAP) != 0: 
                mAPs.append(np.mean(mAP))
            else: mAPs.append(0)
        print("mAP@{}={:.4f}".format(topk, np.mean(mAPs)))
    #segmentation performance
    mIoU=[]
    te_mask = te_mask.cpu().numpy()
    te_mask_pd = te_mask_pd.cpu().numpy()
    for i in range(len(te_mask)):
        iou_score = te_mask[i] == te_mask_pd[i]
        mIoU.append(np.mean(iou_score))
    print("mIoU={:.4f}".format(np.mean(mIoU)))
    #classification performance
    #te_label = te_label.cpu().numpy()
    te_label_pd = te_label_pd.cpu().numpy()
    print ( 'Accuracy: %.6f'%accuracy_score(te_label, te_label_pd))

def main():
    Train() #for training
    Test() #for test

if __name__ == '__main__':
    main()