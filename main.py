import os
import torch.utils.data
from torch.nn import DataParallel
from datetime import datetime


from backbone.cbam import CBAMResNet, CBAMResNet_ae
from margin.ArcMarginProduct import ArcMarginProduct
from margin.MultiMarginProduct import MultiMarginProduct
from margin.CosineMarginProduct import CosineMarginProduct
from margin.InnerProduct import InnerProduct
from utils.logging import init_log
from dataset.casia_webface import CASIAWebFace
from dataset.lfw import LFW
from torch.optim import lr_scheduler
import torch.optim as optim

import time
from eval_lfw import evaluation_10_fold, getFeatureFromTorch
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


train_root = '../CASIA-WebFace' 
train_file_list = 'names_2000.txt'

lfw_test_root = '../Siamese_lfw_pytorch-master/lfw' 
lfw_file_list = 'pairs.txt'

resume =  False  
net_path =  '' 
margin_path =  ''   

device = torch.device('cuda')

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default='Res50_IR', type=str,
                        help="Backbone", )
    parser.add_argument("--save_dir", default='./model', type=str,
                        help="Model Save Directory", )
    parser.add_argument("--batch_size", default=64, type=int,
                        help="Batch size" )
    parser.add_argument("--feature_dim", default=512, type=int,
                        help="Feature_dim" )
    parser.add_argument("--epochs", default=40, type=int,
                        help="Epochs", )
    parser.add_argument("--save_freq", default=300, type=int,
                        help="Saving frequency", )
    return parser


def makeSaveDir(args):
    save_dir = os.path.join(args.save_dir, args.backbone.upper() + '_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    if os.path.exists(save_dir):
        raise NameError('model dir exists!')
    os.makedirs(save_dir)
    logging = init_log(save_dir)
    _print = logging.info
    return save_dir


def myDataLoader(args):
    # dataset loader
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112,112)),
        transforms.ToTensor()
    ])
    augment = transforms.RandomChoice([
        transforms.RandomCrop(size=(64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-180, 180))
    ])
    transform_augment = transforms.Compose([
        transforms.ToPILImage(),
        augment,
        transforms.Resize((112,112)),
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0] 
    ])
    # validation dataset
    trainset = CASIAWebFace(train_root, train_file_list, transform=transform_augment)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=4, drop_last=False)
    # test dataset
    lfwdataset = LFW(lfw_test_root, lfw_file_list, transform=transform)
    lfwloader = torch.utils.data.DataLoader(lfwdataset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=4, drop_last=False)

    return trainset, trainloader, lfwdataset, lfwloader


class MLP(nn.Module):
    def __init__(self, feature_dim=1024, hidden_dim=64, out_dim=1, drop_rate=0.5):
        super(MLP, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.drop_rate = drop_rate
        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.hidden_dim, self.out_dim),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.mlp(input)


def train(args, writer, save_dir, trainset, trainloader, lfwdataset, lfwloader):
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

    margin = MLP(2*args.feature_dim, hidden_dim=64,  out_dim=1, drop_rate=0.5).to(device)

    if resume:
        print('resume the model parameters from: ', net_path, margin_path)
        net.load_state_dict(torch.load(net_path)['net_state_dict'])
        margin.load_state_dict(torch.load(margin_path)['net_state_dict'])

    # define optimizers for different layer
    criterion = torch.nn.BCELoss().to(device)
    optimizer_ft = optim.SGD([
        {'params': net.parameters(), 'weight_decay': 5e-4},
        {'params': margin.parameters(), 'weight_decay': 5e-4}
    ], lr=0.1, momentum=0.9, nesterov=True)

    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[8, 13, 18, 23, 28, 33, 38], gamma=0.2)

    net = net.to(device)
    margin = margin.to(device)

    best_lfw_acc = 0.0
    best_lfw_iters = 0

    total_iters = 0
    total_epoch = args.epochs

    for epoch in range(1, total_epoch + 1):
        exp_lr_scheduler.step()
        # train model
        print('Train Epoch: {}/{} ...'.format(epoch, total_epoch))
        net.train()
        margin.train()

        since = time.time()
        for data in trainloader:
            img_0, img_1, label = data[0].to(device), data[1].to(device), data[2].float().to(device)
            optimizer_ft.zero_grad()

            hidden_0 = net(img_0)
            hidden_1 = net(img_1)
            output = margin(torch.cat((hidden_0, hidden_1), dim=1)).squeeze().to(device)
            batch_loss = criterion(output, label)
            batch_loss.backward()
            optimizer_ft.step()

            # print train information
            if total_iters % 100 == 0:
                
                # current training accuracy
                total = float(label.size(0))
                correct = ((output > 0.5).cpu().int() == label.cpu().int()).float().sum()
                time_cur = (time.time() - since) / 100
                since = time.time()

                print("Iters: {:0>6d}/[{:0>2d}], loss: {:.4f}, train_accuracy: {:.4f}, time: {:.2f} s/iter, learning rate: {}".
                    format(total_iters, epoch, batch_loss.item(), correct/total, time_cur, exp_lr_scheduler.get_lr()[0]))
                writer.add_scalar('Train_Loss/Step', batch_loss, total_iters)
                writer.add_scalar('Train_Acc/Step', correct/total, total_iters)

            # save model
            if total_iters % args.save_freq == 0:
                msg = 'Saving checkpoint: {}'.format(total_iters)
                print(msg)
                
                net_state_dict = net.state_dict()
                margin_state_dict = margin.state_dict()
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                torch.save({
                    'iters': total_iters,
                    'net_state_dict': net_state_dict},
                    os.path.join(save_dir, 'Iter_%06d_net.ckpt' % total_iters))
                torch.save({
                    'iters': total_iters,
                    'net_state_dict': margin_state_dict},
                    os.path.join(save_dir, 'Iter_%06d_margin.ckpt' % total_iters))

            # test accuracy
            if total_iters % args.save_freq == 0:
                print('Testing on LFW')
                with torch.no_grad(): 

                    # test model on lfw
                    net.eval()
                    margin.eval()
                    # getFeatureFromTorch('./result/cur_lfw_result.mat', net, device, lfwdataset, lfwloader)
                    # lfw_accs = evaluation_10_fold('./result/cur_lfw_result.mat')
                    test_loss = []
                    test_output = []
                    test_label = []
                    for lfw_data in lfwloader:
                        img_0, img_1, label = lfw_data[0].to(device), lfw_data[1].to(device), lfw_data[2].float().to(device)
                        hidden_0 = net(img_0)
                        hidden_1 = net(img_1)
                        output = margin(torch.cat((hidden_0, hidden_1), dim=1)).squeeze().to(device)
                        batch_loss = criterion(output, label)

                        test_loss.append(batch_loss.cpu().detach())
                        test_output.extend(output.cpu().detach().numpy().tolist())
                        test_label.extend(label.cpu().detach().numpy().tolist())
                    
                    print('Test Loss: {:.4f}'.format(np.mean(test_loss)))
                    test_size = len(test_output)
                    test_output = np.array(test_output, dtype=float)
                    test_label = np.array(test_label, dtype=float)
                    test_correct = np.sum((test_output > 0.5) == test_label)
                    test_acc = float(test_correct) / float(test_size)
                    print('Correct: %d / Total: %d' % (test_correct, test_size))
                    print('LFW Ave Accuracy: {:.4f}'.format((test_correct / test_size) * 100))
                    writer.add_scalar('Test_Acc/Step', test_acc, total_iters)

                    if best_lfw_acc <= test_acc * 100:
                        best_lfw_acc = test_acc * 100
                        best_lfw_iters = total_iters

            net.train()
            margin.train()
            total_iters += 1

    print('Finally Best Accuracy: LFW: {:.4f} in iters: {}'.format(best_lfw_acc, best_lfw_iters))
    print('finishing training')


if __name__ == '__main__':
    args = get_argparse().parse_args()
    save_dir = makeSaveDir(args)
    trainset, trainloader, lfwdataset, lfwloader = myDataLoader(args)
    writer = SummaryWriter(comment=args.backbone)
    train(args, writer, save_dir, trainset, trainloader, lfwdataset, lfwloader)


