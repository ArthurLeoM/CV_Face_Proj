import train
import os
import torch.utils.data
from torch.nn import DataParallel
from datetime import datetime

from backbone.cbam import CBAMResNet, CBAMResNet_ae
from utils.logging import init_log
from dataset.casia_webface import CASIAWebFace
from dataset.lfw import LFW_test, LFW_val, LFW_train
from torch.optim import lr_scheduler
import torch.optim as optim

import time
import numpy as np
import torchvision.transforms as transforms
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from PIL import Image
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

device = torch.device('cuda')
lfw_root = '../Siamese_lfw_pytorch-master/lfw' 
lfw_file_list = 'pairs.txt'

def test(testloader, args, save_dir, num_iter):
    if args.backbone == 'Res50_IR':
        net = CBAMResNet(50, feature_dim=args.feature_dim, mode='ir').to(device)
    elif args.backbone == 'SERes50_IR':
        net = CBAMResNet(50, feature_dim=args.feature_dim, mode='ir_se').to(device)
    elif args.backbone == 'Res100_IR':
        net = CBAMResNet(100, feature_dim=args.feature_dim, mode='ir').to(device)
    elif args.backbone == 'SERes100_IR':
        net = CBAMResNet(100, feature_dim=args.feature_dim, mode='ir_se').to(device)
    else:
        print(args.backbone, ' is not available!')

    margin = train.MLP(2*args.feature_dim, hidden_dim=64,  out_dim=1, drop_rate=0.5).to(device)

    net_param = torch.load(os.path.join(save_dir, 'Iter_%06d_net.ckpt' % num_iter))['net_state_dict']
    net.load_state_dict(net_param)
    margin_param = torch.load(os.path.join(save_dir, 'Iter_%06d_margin.ckpt' % num_iter))['net_state_dict']
    margin.load_state_dict(margin_param)


    with torch.no_grad(): 
        # test model on lfw
        net.eval()
        margin.eval()
        # getFeatureFromTorch('./result/cur_lfw_result.mat', net, device, lfwdataset, lfwloader)
        # lfw_accs = evaluation_10_fold('./result/cur_lfw_result.mat')
        test_label = []
        featureLs = None
        featureRs = None
        for lfw_data in testloader:
            img_0, img_1, label = lfw_data[0].to(device), lfw_data[1].to(device), lfw_data[2].float().to(device)
            hidden_0 = net(img_0).cpu().numpy()
            hidden_1 = net(img_1).cpu().numpy()
            test_label.extend(label.cpu().detach().numpy().tolist())

            if featureLs is None:
                featureLs = hidden_0
            else:
                featureLs = np.concatenate((featureLs, hidden_0), 0)
            
            if featureRs is None:
                featureRs = hidden_1
            else:
                featureRs = np.concatenate((featureRs, hidden_1), 0)
        
        test_size = len(test_label)
        test_label = np.array(test_label)
        mu = np.mean(np.concatenate((featureLs, featureRs), 0), 0)
        mu = np.expand_dims(mu, 0)
        featureLs = featureLs / np.expand_dims(np.sqrt(np.sum(np.power(featureLs, 2), 1)), 1)
        featureRs = featureRs / np.expand_dims(np.sqrt(np.sum(np.power(featureRs, 2), 1)), 1)

        scores = np.sum(np.multiply(featureLs, featureRs), 1)
        threshold = train.getThreshold(scores, test_label, 10000)
        test_acc, p, n = train.getAccuracy(scores, test_label, threshold)
        print('Correct: %d / Total: %d' % (p+n, test_size), flush=True)
        print('LFW Ave Accuracy: {:.4f}'.format(test_acc), flush=True)

if __name__ == '__main__':
    args = train.get_argparse().parse_args()
    _, _, _, _, testset, testloader = train.myDataLoader(args)
    test(testloader, args, save_dir='./model/RES50_IR_20201206_015114', num_iter=34500)
