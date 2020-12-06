import os
import torch.utils.data
from torch.nn import DataParallel
from datetime import datetime

from backbone.cbam import CBAMResNet, CBAMResNet_ae
from utils.logging import init_log
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


lfw_root = '../Siamese_lfw_pytorch-master/lfw' 
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
    parser.add_argument("--save_freq", default=500, type=int,
                        help="Saving frequency", )
    return parser


def getAccuracy(scores, flags, threshold):
    p = np.sum(scores[flags == 1] > threshold)
    n = np.sum(scores[flags == 0] < threshold)
    return 1.0 * (p + n) / len(scores), p, n


def getThreshold(scores, flags, thrNum):
    accuracys = np.zeros((2 * thrNum + 1, 1))
    thresholds = np.arange(-thrNum, thrNum + 1) * 0.1 / thrNum
    for i in range(2 * thrNum + 1):
        accuracys[i] = getAccuracy(scores, flags, thresholds[i])[0]
    max_index = np.squeeze(accuracys == np.max(accuracys))
    bestThreshold = np.mean(thresholds[max_index])
    return bestThreshold


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

    # validation dataset
    trainset = LFW_train(lfw_root, lfw_file_list, transform=None)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8,
                                            shuffle=True, num_workers=4, drop_last=False)
    # val dataset
    valset = LFW_val(lfw_root, lfw_file_list, transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=4, drop_last=False)
    # test dataset
    testset = LFW_test(lfw_root, lfw_file_list, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=4, drop_last=False)

    return trainset, trainloader, valset, valloader, testset, testloader


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


def train(args, writer, save_dir, trainset, trainloader, valset, valloader):
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

    augment = transforms.RandomChoice([
        transforms.RandomCrop(size=(128, 128)),
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

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112,112)),
        transforms.ToTensor()
    ])

    # define optimizers for different layer
    criterion = torch.nn.BCELoss().to(device)
    net_optimizer_ft = optim.SGD([
        {'params': net.parameters(), 'weight_decay': 5e-4},
    ], lr=1e-2, momentum=0.9, nesterov=True)
    margin_optimizer_ft = optim.SGD([
        {'params': margin.parameters()}
    ], lr=1e-5, momentum=0.9, nesterov=True)

    exp_lr_scheduler_net = lr_scheduler.MultiStepLR(net_optimizer_ft, milestones=[8, 18, 28, 38], gamma=0.2)
    exp_lr_scheduler_margin = lr_scheduler.MultiStepLR(margin_optimizer_ft, milestones=[8, 18, 28, 38], gamma=5)

    net = net.to(device)
    margin = margin.to(device)

    best_lfw_acc = 0.0
    best_lfw_iters = 0

    total_iters = 0
    total_epoch = args.epochs

    for epoch in range(1, total_epoch + 1):
        exp_lr_scheduler_net.step()
        exp_lr_scheduler_margin.step()
        # train model
        print('Train Epoch: {}/{} ...'.format(epoch, total_epoch))
        net.train()
        margin.train()

        since = time.time()
        for data in trainloader:
            sample_imgs = data    #np array
            img_0 = []
            img_1 = []
            label = []
            for i in sample_imgs:
                # pos img_0
                for j in range(2):
                    img_0.append(transform(i.numpy()).numpy())
                for j in range(2):
                    img_0.append(transform_augment(i.numpy()).numpy())
                # neg img_0
                for j in range(2):
                    img_0.append(transform(i.numpy()).numpy())
                for j in range(2):
                    img_0.append(transform_augment(i.numpy()).numpy())
                
                # pos img_1
                for j in range(4):
                    img_1.append(transform_augment(i.numpy()).numpy())
                # neg img_1
                for j in range(2):
                    idx = np.random.randint(0, len(trainset.train_name))
                    tmp_img = trainset.loader(os.path.join(lfw_root, trainset.train_name[idx]))
                    img_1.append(transform(tmp_img).numpy())
                for j in range(2):
                    idx = np.random.randint(0, len(trainset.train_name))
                    tmp_img = trainset.loader(os.path.join(lfw_root, trainset.train_name[idx]))
                    img_1.append(transform_augment(tmp_img).numpy())

                #labels
                label.extend([1, 1, 1, 1, 0, 0, 0, 0])

            img_0 = torch.tensor(np.array(img_0), dtype=torch.float).to(device)
            img_1 = torch.tensor(np.array(img_1), dtype=torch.float).to(device)
            label = torch.tensor(np.array(label), dtype=torch.float).to(device)

            #print(img_0.shape, img_1.shape, label.shape)

                
            net_optimizer_ft.zero_grad()
            margin_optimizer_ft.zero_grad()

            hidden_0 = net(img_0)
            hidden_1 = net(img_1)
            output = margin(torch.cat((hidden_0, hidden_1), dim=1)).squeeze().to(device)
            #print(label, output)
            batch_loss = criterion(output, label).to(device)
            batch_loss.backward()
            net_optimizer_ft.step()
            margin_optimizer_ft.step()

            # print train information
            if total_iters % 50 == 0:
                
                # current training accuracy
                total = float(label.size(0))
                correct = ((output > 0.5).cpu().int() == label.cpu().int()).float().sum()
                time_cur = (time.time() - since) / 100
                since = time.time()

                print("Iters: {:0>6d}/[{:0>2d}], loss: {:.4f}, train_accuracy: {:.4f}, time: {:.2f} s/iter, learning rate: {}".
                    format(total_iters, epoch, batch_loss.item(), correct/total, time_cur, exp_lr_scheduler_net.get_lr()[0]), flush=True)
                writer.add_scalar('Train_Loss/Step', batch_loss, total_iters)
                writer.add_scalar('Train_Acc/Step', correct/total, total_iters)


            # Val accuracy
            if total_iters % args.save_freq == 0:
                print('Validating on LFW')
                with torch.no_grad(): 

                    # test model on lfw
                    test_label = []
                    featureLs = None
                    featureRs = None
                    for lfw_data in valloader:
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
                    threshold = getThreshold(scores, test_label, 10000)
                    test_acc, p, n = getAccuracy(scores, test_label, threshold)
                    print('Correct: %d / Total: %d' % (p+n, test_size), flush=True)
                    print('LFW_Val Ave Accuracy: {:.4f}'.format(test_acc), flush=True)
                    writer.add_scalar('Val_Acc/Step', test_acc, total_iters)

                    if best_lfw_acc <= test_acc * 100:
                        best_lfw_acc = test_acc * 100
                        best_lfw_iters = total_iters

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

            # net.train()
            # margin.train()
            total_iters += 1

    print('Finally Best Accuracy: LFW: {:.4f} in iters: {}'.format(best_lfw_acc, best_lfw_iters))
    print('finishing training')


if __name__ == '__main__':
    args = get_argparse().parse_args()
    save_dir = makeSaveDir(args)
    trainset, trainloader, valset, valloader, _, _ = myDataLoader(args)
    writer = SummaryWriter(comment=args.backbone)
    train(args, writer, save_dir, trainset, trainloader, valset, valloader)
    


