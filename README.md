# Resnet50 and drn_a_50 wiht MNIST fashion

## Pytorch 4.1 is suppoted on branch 0.4 now.
## Support Arc:
* Resnet50 [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
* Drn_a_50 [Dilated Residual Networks](https://arxiv.org/abs/1705.09914)
## Support Code and model:
* Resnet50 [https://download.pytorch.org/models/resnet50-19c8e357.pth]
* Drn_a_50 [https://github.com/fyu/drn]

### Fashion MNIST Test
| System                                   |  *top1*  |  *top5* | **FPS** (GeoForce 1080) |
| :--------------------------------------- | :------: |:-------:| :-----------------------: |
| Resnet50                                 |   92.48   | 99.94  |     **150 (1080Ti)**      |
| Drn_a_50                                 |   78.8   |          |       120 (1080Ti)        |

### Contents
1. [Installation](#installation)
2. [Datasets](#datasets)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Models](#models)

## Installation
- Install [PyTorch-0.4.1](http://pytorch.org/) by selecting your environment on the website and running the appropriate command.
- Clone this repository. This repository is mainly based on [drn](https://github.com/fyu/drn) and [fashion-mnist](https://github.com/zalandoresearch/fashion-mnist), a huge thank to them.
  * Note: We currently only support Python 3+. 
  
## Datasets
The dataset is downloaded from the links below, which is stored in the same format as the original MNIST data.

| name                                   |  content |  examples | link |
| :--------------------------------------- | :------: |:-------:| :-----------------------: |
|train-images-idx3-ubyte.gz | training set images |60,000|http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz  | 
| train-labels-idx3-ubyte.gz | training set labels | 60,000|http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz |
|t10k-images-idx3-ubyte.gz | testing set images |10,000|http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz |
|t10k-labels-idx3-ubyte.gz | testing set labels |10,000|http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz |

The data are processed in by instructions in https://github.com/zalandoresearch/fashion-mnist, and the processed dataset in this repo is located in './data/processed'.

* Note: In this repo, the training set is divided into two parts: one is for training with size 50,000 and the other is for evaluating with size 10,000. The processed evaluating dataset is located in '/data/processed/val' with the name 'test.pth', while the testing dataset is located in '/data/processed/test' with the name 'test.pth'.
 
## Training
```
- To train drn_a_50 Net using the train script simply specify the parameters listed in `train_drn.py` as a flag or manually change them. To train resnet50 is in the same way except using the 'train-resnet50.py' script.
```Shell
python train-drn.py 
```
   * Note:
        --patience      # early stopping
        --batch_size    # batch size
        --nepochs       # max epochs
        --nworkers      # number of workers
        --seed          # random seed
        --data          #FashionMNIST or MNIST
          
## Evaluation
- You can test the drn_a_50 Net with the code below, and to test resnet50 is as the same except using 'test-resnet50.py'.  

```Shell
python test-drn.py 
```
  * Note:
        --data          #FashionMNIST or MNIST
        --batch_size    #Batch size
        -p              #print frequency (default: 10)
        
