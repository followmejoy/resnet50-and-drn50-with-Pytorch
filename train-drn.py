### Train dilated resnet-50 on Fashion MINIST 
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

parser = argparse.ArgumentParser(description='Training of dilated resnet50 on FashionMNIST')
parser.add_argument("--model", type=str, default='drn_a_50', help="model:drn_a_50")
parser.add_argument("--patience", type=int, default=5, help="early stopping patience")
parser.add_argument('--epochs', default=90, type=int, metavar='N',help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',help='manual epoch number (useful on restarts)')
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument('--lr', '--learning-rate', default=10e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument("--nworkers", type=int, default=8, help="number of workers")
parser.add_argument("--seed", type=int, default=1, help="random seed")
parser.add_argument("--data", type=str, default='FashionMNIST', help="FashionMNIST")
parser.add_argument('--step-ratio', dest='step_ratio', type=float, default=0.1)
parser.add_argument('--print-freq', '-p', default=10, type=int,metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default=False, type=bool, help='resume from a checkpoint')
parser.add_argument('--resume_epoch', default=0, type = str, help='resume from the epoch checkpoint')
parser.add_argument("--save_folder", type=str, default='saved-weights', help="folder to save the trained models")
parser.add_argument('--visdom', default=False, type=bool, help='Use visdom for loss visualization')


args = parser.parse_args()

# Set up the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Training on {}'.format(device))

# Set seeds. If using numpy this must be seeded too.
torch.manual_seed(args.seed)
if device== 'cuda:0':
    torch.cuda.manual_seed(args.seed)
    
if args.visdom:
    import visdom
    viz = visdom.Visdom()
    
# Setup folders for saved models and logs
if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
if not os.path.exists('logs/'):
    os.mkdir('logs/')

# Setup folders. Each run must have it's own folder. Creates
# a logs folder for each model and each run.
out_dir = 'logs/{}'.format(args.model)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
current_dir = '{}/train-val-{}'.format(out_dir, time.strftime('_%m-%d-%H-%M-%S', time.localtime(time.time())))
os.mkdir(current_dir)
# creat log file
logfile = open('{}/log.txt'.format(current_dir), 'w')
print(args, file=logfile)
         
def run_training(args):
    
    print(args, file=logfile)
    
    # creat model
    model = drn.__dict__[args.model]().to(device)

    best_prec1 = 0
    start_epoch = args.start_epoch
    # optionally resume from a checkpoint, default the latest one.
    if args.resume:
        resume_path = os.path.join(args.save_folder, args.model+'_checkpoint_epoch_{}.pth.tar'.format(args.resume_epoch))
        if os.path.exists(resume_path):
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path)
            start_epoch = args.resume_epoch
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])      
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_path, args.resume_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(args.save_folder))
    
    cudnn.benchmark = True
    #data load
    # Define transforms.
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Create dataloaders. Use pin memory if cuda.
    if args.data == 'FashionMNIST':
        trainset = datasets.FashionMNIST('./data', train=True, download=True, transform=train_transforms)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.nworkers)
        valset = datasets.FashionMNIST('./data', train=False, transform=val_transforms)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                shuffle=True, num_workers=args.nworkers)
        print('Training on FashionMNIST')
    else:
        print('Only Training on FashionMNIST')

    epoch_size = len(trainset)//args.batch_size
        
    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(args, optimizer, epoch)

        # train for one epoch
        train(args, train_loader, model, criterion, optimizer, epoch,  epoch_size)

        # evaluate on validation set
        prec1 = validate(args, val_loader, model, criterion)

        patience = args.patience
        # remember best prec@1 and save checkpoint 
        is_best = prec1 > best_prec1
        if is_best:
            best_prec1 = prec1
            if args.resume == False:
                patience = args.patience
            utils.save_model({
                'arch': args.model,
                'best_prec1': best_prec1,
                'state_dict': model.state_dict()
            }, '{}/{}_checkpoint_epoch_{}.pth.tar'.format(args.save_folder, args.model, epoch))
        else:
            patience -= 1
            if patience == 0:
                print('Run out of patience!')
                break    
            

def train(args, train_loader, model, criterion, optimizer, epoch,  epoch_size):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

            
    writeFile = open('{}/{}-stats-train.csv'.format(current_dir, args.model), 'a')
    writer = csv.writer(writeFile)
    writer.writerow(['Epoch', 'Time', 'Data', 'Loss', 'prec@1', 'prec@5'])
    if args.visdom:
        vis_title = 'DRN.PyTorch on ' + args.data
        vis_legend = ['Loss', 'prec@1', 'prec@5']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)        
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)       
    
    end = time.time()
    for i, (input_var, target) in enumerate(train_loader):
        
        if args.visdom and i != 0 and (i % epoch_size == 0):
            update_vis_plot(epoch, losses.val, top1.val, top5.val, epoch_plot, None,
                            'append', epoch_size)
            
        # measure data loading time
        data_time.update(time.time() - end)


        m=torch.nn.Upsample(scale_factor=8, mode='nearest')
        input_var = m(input_var)
        input_var = torch.autograd.Variable(input_var)
        target = torch.autograd.Variable(target)
        
        input_var, target = input_var.to(device), target.to(device)
        # compute output
        output = model(input_var)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input_var.size(0))
        top1.update(prec1[0], input_var.size(0))
        top5.update(prec5[0], input_var.size(0))
 
        if args.visdom :
            update_vis_plot(i, losses.val, top1.val, top5.val, iter_plot, epoch_plot,'append')   
            
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            # Write to csv file
            writer.writerow([i, batch_time.val, data_time, losses.val,top1.val, top5.val])
    writeFile.close()            
def validate(args, val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    writeFile = open('{}/{}-stats-val.csv'.format(current_dir, args.model), 'a')
    writer = csv.writer(writeFile)
    writer.writerow(['Epoch', 'Time', 'Loss', 'prec@1', 'prec@5'])
    
    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input_var, target) in enumerate(val_loader):
        
        m=torch.nn.Upsample(scale_factor=8, mode='nearest')
        input_var = m(input_var)
        input_var = torch.autograd.Variable(input_var)
        target = torch.autograd.Variable(target)
        
        input_var, target = input_var.to(device), target.to(device)
        # compute output
        output = model(input_var)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input_var.size(0))
        top1.update(prec1[0], input_var.size(0))
        top5.update(prec5[0], input_var.size(0))

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
    writeFile.close()    
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


def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (args.step_ratio ** (epoch // 30))
    print('Epoch [{}] Learning rate: {}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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

def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loss, top1, top5, window1, window2,update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loss, top1, top5]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loss, top1, top5]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )



if __name__ == '__main__':
    
    run_training(args)


