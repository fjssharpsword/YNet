import os
import cv2
import time
import sys
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import PIL.ImageOps
import numpy as np
import torchvision.transforms as transforms
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, average_precision_score
#self-defined
from dataset.origa import get_train_dataloader, get_test_dataloader
#from dataset.jsrt import get_train_dataloader, get_test_dataloader
from util.Evaluation import compute_AUCs, compute_fusion
from net.YNet import YNet, YNetLoss, CircleLoss

def main_fig1():
    background = Image.open('/data/fjsdata/JSRT-CXR/images/JPCLN002.png').resize((1024,1024)) #cv2.imread('/data/fjsdata/JSRT-CXR/images/JPCLN001.png')
    overlay = Image.open('/data/fjsdata/JSRT-CXR/masks/left_lung/JPCLN002.gif') #cv2.imread('/data/fjsdata/JSRT-CXR/masks/left_lung/JPCLN001.gif')

    background = np.asarray(background.convert('RGB'))
    overlay = np.asarray(overlay.convert('RGB'))

    added_image = cv2.addWeighted(background, 0.5, overlay,0.5,0)
    fig, ax = plt.subplots(1)# Create figure and axes
    ax.imshow(added_image)
    #rect = patches.Rectangle((x2, y2), w, h, linewidth=2, edgecolor='r', facecolor='none')# Create a Rectangle patch
    #ax.add_patch(rect)# Add the patch to the Axes
    ax.axis('off')
    ax.set_title('Benign nodule of chest X-ray')
    fig.savefig('/data/pycode/YNet/imgs/JPCLN002_mask.png')

# generate class activation mapping for the predicted classed
def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 224x224
    size_upsample = (self.TRAN_CROP, self.TRAN_CROP)
    bz, nc, h, w = feature_conv.shape

    cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc,h*w)))
        #cam = weight_softmax[class_idx]*(feature_conv.reshape((nc,h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, size_upsample)
    return cam_img

def main_fig3():

    # Visualize feature maps
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # normalizing the output
    def normalize_output(img):
        img = img - img.min()
        img = img / img.max()
        return img

    transform_seq_image = transforms.Compose([
            transforms.Resize((256,256)),#256
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            ])

    transform_seq_mask = transforms.Compose([
            transforms.Resize((256,256))
            ])

    image = Image.open('/data/fjsdata/JSRT-CXR/images/JPCLN001.png').convert('RGB')
    image = transform_seq_image(image)
    image = image.unsqueeze(0).cuda()

    mask = Image.open('/data/fjsdata/JSRT-CXR/masks/left_lung/JPCLN001.gif').convert('RGB')
    mask = np.asarray(transform_seq_mask(mask))
    #load model
    model = YNet(n_classes=2, n_masks=3, code_size=64).cuda()
    CKPT_PATH = '/data/pycode/YNet/model/best_model.pkl'
    checkpoint = torch.load(CKPT_PATH)
    model.load_state_dict(checkpoint) #strict=False
    print("=> loaded Image model checkpoint: "+CKPT_PATH)
    
    #register conv8/layer5, 256*8*8
    model.conv8.register_forward_hook(get_activation('conv8'))  
    
    #forward
    h_feat, h_cls, s_mask, c_feat, c_cls = model(image)
    #extract conv feature
    feature = activation['conv8'].squeeze()
    feature = torch.mean(feature, dim=0).cpu().numpy()
    feature = normalize_output(feature)
    feature = np.uint8(255 * feature)
    #plot
    height, width = mask.shape[0], mask.shape[1]
    featuremap = cv2.applyColorMap(cv2.resize(feature, (width, height)), cv2.COLORMAP_JET)
    overlay_img = cv2.addWeighted(mask, 0.7, featuremap, 0.3, 0)
    fig, ax = plt.subplots(1)# Create figure and axes
    ax.imshow(overlay_img)
    ax.axis('off')
    fig.savefig('/data/pycode/YNet/imgs/JPCLN001_fea_overlay.png')

def main_fig6():
    #query, top1, top2, top3, top4, top5 =  JPCLN001, JPCLN003, JPCLN005, JPCLN009, JPCLN010, JPCLN013 (YNet)
    #query, top1, top2, top3, top4, top5 =  JPCLN001, JPCLN028, JPCLN030, JPCLN062, JPCLN035, JPCLN075 (DRH)
    query = Image.open('/data/fjsdata/JSRT-CXR/images/JPCLN075.png').resize((1024,1024))
    query_mask = Image.open('/data/fjsdata/JSRT-CXR/masks/left_lung/JPCLN075.gif')
    query = np.asarray(query.convert('RGB'))
    query_mask = np.asarray(query_mask.convert('RGB'))

    added_image = cv2.addWeighted(query, 0.5, query_mask,0.5,0)
    fig, ax = plt.subplots(1)# Create figure and axes
    ax.imshow(added_image)
    ax.axis('off')
    fig.savefig('/data/pycode/YNet/imgs/JPCLN075_top5.png')

def main_fig8():
    print('********************load data********************')
    dataloader_train = get_train_dataloader(batch_size=36, shuffle=False, num_workers=6)
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    model = YNet(n_classes=2, n_masks=3, code_size=64).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    lr_scheduler_model = lr_scheduler.StepLR(optimizer , step_size = 10, gamma = 1)
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    ce_loss  = nn.CrossEntropyLoss() #for segmentation
    cl_loss = CircleLoss() #for classification
    ynet_loss = YNetLoss().cuda() #for unified loss
    print('********************load model succeed!********************')

    print('********************begin training!********************')
    best_loss = float('inf')
    circle_loss, cross_loss, sum_loss, couple_loss = [], [], [], []
    for epoch in range(50):
        since = time.time()
        print('Epoch {}/{}'.format(epoch+1 , 50))
        print('-' * 10)
        model.train()  #set model to training mode
        train_loss = []
        circle_loss_ep, ce_loss_ep, sum_loss_ep, couple_loss_ep = [], [], [], []
        with torch.autograd.enable_grad():
            for batch_idx, (image, mask, label) in enumerate(dataloader_train):
                optimizer.zero_grad()
                #forward
                var_image = torch.autograd.Variable(image).cuda()
                var_mask = torch.autograd.Variable(mask).cuda()
                var_label = torch.autograd.Variable(label).cuda()
                h_feat, h_cls, s_mask, c_feat, c_cls = model(var_image)
                #backward
                mask_loss = ce_loss(s_mask, var_mask)
                cls_loss =  cl_loss(c_feat, var_label) #ce_loss(c_cls, y_batch)
                loss_tensor = ynet_loss(cls_loss, mask_loss)
                loss_tensor.backward() 
                optimizer.step()
                #print([x.grad for x in optimizer.param_groups[0]['params']])
                sys.stdout.write('\r Epoch: {} / Step: {} : train loss = {}'.format(epoch+1, batch_idx+1, float('%0.6f'%loss_tensor.item()) ))
                sys.stdout.flush()       
                train_loss.append(loss_tensor.item()) 

                circle_loss_ep.append(cls_loss.item())
                ce_loss_ep.append(mask_loss.item())
                sum_loss_ep.append(cls_loss.item() + mask_loss.item())
                couple_loss_ep.append(loss_tensor.item()) 
        lr_scheduler_model.step()  #about lr and gamma
        print("\r Eopch: %5d train loss = %.6f" % (epoch + 1, np.mean(train_loss))) 
        circle_loss.append(np.mean(circle_loss_ep))
        cross_loss.append(np.mean(ce_loss_ep))
        sum_loss.append(np.mean(sum_loss_ep))
        couple_loss.append(np.mean(couple_loss_ep))
        
        time_elapsed = time.time() - since
        print('Training epoch: {} completed in {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60 , time_elapsed % 60))

    x_axix = range(50)
    cls_loss_pt = plt.plot(x_axix,circle_loss,label='Circle loss of the R-MAC branch', color='y')
    seg_loss_pt = plt.plot(x_axix,cross_loss,label='Cross-entropy loss of the FPN branch', color='g')
    sum_loss_pt = plt.plot(x_axix,sum_loss,label='Sum of two losses', color='b', linestyle='-')
    couple_loss_pt = plt.plot(x_axix,couple_loss,label='Coupled loss of Y-Net', color='r', linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('Training loss')
    plt.title('Fundus Dataset')
    plt.legend()
    plt.savefig('/data/pycode/YNet/imgs/Fundus_loss.jpg')

def main_APCurve():
    print('********************load data********************')
    dataloader_test = get_test_dataloader(batch_size=36, shuffle=False, num_workers=6)
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    model = model = YNet(n_classes=2, n_masks=3, code_size=64).cuda()
    CKPT_PATH = '/data/pycode/YNet/model/best_model.pkl'
    checkpoint = torch.load(CKPT_PATH)
    model.load_state_dict(checkpoint) #strict=False
    print("=> loaded Image model checkpoint: "+CKPT_PATH)
    
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    model.eval() #turn to test mode
    print('******************** load model succeed!********************')

    te_label = torch.FloatTensor().cuda()
    te_label_pd = torch.FloatTensor().cuda()
    with torch.autograd.no_grad():
        for batch_idx, (image, mask, label) in enumerate(dataloader_test):
            te_label = torch.cat((te_label, label.cuda()), 0)
            var_image = torch.autograd.Variable(image).cuda()
            _, h_cls, _, _, c_cls = model(var_image)
            c_cls = F.log_softmax(c_cls,dim=1) 
            #c_cls = c_cls.max(1,keepdim=True)[0] #return probability
            #c_cls = c_cls.max(1,keepdim=True)[1] #return index
            te_label_pd = torch.cat((te_label_pd, c_cls.data), 0)
            sys.stdout.write('\r test indexing process: = {}'.format(batch_idx+1))
            sys.stdout.flush()

    print('******* begin evaluating!*********')
    #classification performance
    te_label = te_label.cpu().numpy()
    te_label_pd = te_label_pd.cpu().numpy()[:,0]
    #print ( 'Accuracy: %.6f'%accuracy_score(te_label, te_label_pd))

    #for evaluation   
    ap = average_precision_score(te_label, te_label_pd)
    print("\r Test average precision = %.4f" % (ap)) 

    #plot precision_recall_curve, precision = tp/(tp+fp), recall = sen =  tp /(tp+fn)
    precision, recall, thresholds = precision_recall_curve(te_label.ravel(), te_label_pd.ravel())
    #plot and save
    plt.plot(recall, precision, c = 'r', ls = '--', label = u'AP={:.4f}'.format(ap))
    plt.plot((1, 0), (0, 1), c = '#808080', lw = 1, ls = '--', alpha = 0.7)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.2))
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(b=True, ls=':')
    plt.legend(loc='lower left')
    plt.title('PR Curve of the Fundus Dataset')
    plt.savefig('/data/pycode/YNet/imgs/Fundus_PRCurve.jpg')


if __name__ == '__main__':
    #main_fig1()
    #main_fig3()
    #main_fig6()
    #main_fig8()
    main_APCurve()
