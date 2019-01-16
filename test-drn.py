### Test drn model on FashionMNIST

import argparse
import shutil
import time

import numpy as np
import os
from os.path import exists, split, join, splitext

import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import utils
import argparse
import csv
import mnist_reader
from tqdm import tqdm
from scipy import misc

import drn

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='drn_a_50', help="model")
parser.add_argument("--data", type=str, default='FashionMNIST', help="FashionMNIST")
parser.add_argument("--seed", type=int, default=1, help="random seed")
parser.add_argument("--batch_size", type=int, default=4, help="batch size")
parser.add_argument('--print-freq', '-p', default=10, type=int,metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default=True, type=bool, help='resume from a checkpoint,default：latest checkpoint ')
parser.add_argument('--resume_epoch', default=14, type = str, help='resume from the epoch checkpoint')
parser.add_argument("--save_folder", type=str, default='saved-weights', help="folder to save the trained models")


args = parser.parse_args()

# Set up the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Testing on {}'.format(device))

## Set seeds. If using numpy this must be seeded too.
torch.manual_seed(args.seed)
if device== 'cuda:0':
    torch.cuda.manual_seed(args.seed)

# Setup folders for saved models and logs
if not os.path.exists(args.save_folder):
    print('No folder that saves the trained models!')
if not os.path.exists('logs/'):
    os.mkdir('logs/')

# Setup folders. Each run must have it's own folder. Creates
# a logs folder for each model and each run.
out_dir = 'logs/{}'.format(args.model)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

current_dir= '{}/test-{}'.format(out_dir,args.model) + time.strftime('_%m-%d-%H-%M-%S', time.localtime(time.time()))
os.mkdir(current_dir)
logfile = open('{}/log.txt'.format(current_dir), 'w')
print(args, file=logfile)

def test_model(args):
    # create model
    model = drn.__dict__[args.model]().to(device)

    if args.resume:
        resume_path = os.path.join(args.save_folder, args.model+'_checkpoint_epoch_{}.pth.tar'.format(args.resume_epoch))
        if os.path.exists(resume_path):
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path)
            model.load_state_dict(checkpoint['state_dict'])        
            print("=> loaded checkpoint '{}' (epoch {})" .format(resume_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.save_folder))

    cudnn.benchmark = True

    # Data loading code
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Create dataloaders. Use pin memory if cuda.
    testset = datasets.FashionMNIST('./data', train=False, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(testset, args.batch_size,
                            shuffle=False, num_workers=0)
    print('Testing on FashionMNIST')


    criterion = nn.CrossEntropyLoss().cuda()

    validate(args, test_loader, model, criterion)
            
            
def validate(args, val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    
    writeFile = open('{}/{}-stats-test.csv'.format(current_dir, args.model), 'a')
    writer = csv.writer(writeFile)
    writer.writerow(['Test', 'Time', 'Loss', 'prec@1', 'prec@5']) 
    
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        
        m=torch.nn.Upsample(scale_factor=8, mode='nearest')
        input=m(input)
        input, target = input.to(device), target.to(device)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))            
            # Write to csv file
            writer.writerow([i, batch_time.val, losses.val,top1.val, top5.val])
        
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))


    return top1.avg



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



if __name__ == '__main__':
    
    test_model(args)


