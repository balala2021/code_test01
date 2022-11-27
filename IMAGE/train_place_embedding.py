from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms, utils
import torchvision.transforms.functional as TF

from tqdm import tqdm
import numpy as np
import json
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
from os.path import join, exists
import copy
import random
from collections import OrderedDict
from sklearn.metrics import r2_score

from utils.image_dataset import PlaceImagePairDataset
from mymodels.skip_gram import PlaceImageSkipGram

# Configuration
#os.environ['TORCH_HOME'] = '/home/ubuntu/projects/TorchModelZoo'

data_dir = "/data/lijinlin/data/居住区visual"
train_path_list_path = '/data/lijinlin/M3G/data/5899小区图像pair-2-train.pkl'
val_path_list_path = '/data/lijinlin/M3G/data/5899小区图像pair-2-test.pkl'


CNN_model_path = '/home/ubuntu/projects/checkpoint/inception_v3_google-1a9a5a14.pth'

old_ckpt_path = None
ckpt_save_dir = 'checkpoint/beijing'


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model_name = 'sv_embedding'
embedding_dim = 200
threshold = 0.5
trainable_params = None
return_best = True
if_early_stop = True
#input_size = 299
input_size =256
learning_rate = [0.0005]
weight_decay = 0.0
batch_size = 16
num_epochs = 21
lr_decay_rate = 0.7
lr_decay_epochs = 11
early_stop_epochs = 11
save_epochs = 5
margin = [5]


def RandomRotationNew(image):
    angle = random.choice([0, 90, 180, 270])
    #随机采样，angle =0？
    image = TF.rotate(image, angle)
    return image

def metrics(stats):
    """
    评价函数
    Self-defined metrics function to evaluate and compare models
    stats: {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}
    return: must be a single number """
    accuracy = (stats['T'] + 0.00001) * 1.0 / (stats['T'] + stats['F'] + 0.00001)
    print(accuracy)
    return accuracy
    # return (stats['TP'] + stats['TN']) * 1.0 / (stats['TP'] + stats['TN'] + stats['FN'] + stats['FP'])


def train_embedding(model, model_name, dataloaders, criterion, optimizer, metrics, num_epochs, threshold=0.5,
                verbose=True, return_best=True, if_early_stop=True, early_stop_epochs=10, scheduler=None,
                save_dir=None, save_epochs=5,dist=None):
    since = time.time()
    training_log = dict()
    embs =[]
    training_log['train_loss_history'] = []
    training_log['val_loss_history'] = []
    training_log['val_metric_value_history'] = []
    training_log['current_epoch'] = -1
    current_epoch = training_log['current_epoch'] + 1

    best_model_wts = copy.deepcopy(model.state_dict()) #对应model的每一层可训练的weights及偏置等
    best_optimizer_wts = copy.deepcopy(optimizer.state_dict()) #优化器的
    best_log = copy.deepcopy(training_log)

    best_metric_value = 0.
    nodecrease = 0  # to count the epochs that val loss doesn't decrease
    early_stop = False

    for epoch in range(current_epoch, current_epoch + num_epochs):
        if verbose:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            stats = {'T': 0, 'F': 0}

            # Iterate over data.
            #锚采样和负采样数据
            for anc_images, pos_images, fips in tqdm(dataloaders[phase],ascii=True):
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    print(anc_images)
                    anc_images = anc_images.to(device) #anc_images
               	    pos_images = pos_images.to(device)  #pos_images？
                    outputs1=model(anc_images)
                    outputs2=model(pos_images)
                     # zero the parameter gradients
                    optimizer.zero_grad()
                    # Get model outputs and calculate loss
                    distance1=dist(outputs1,outputs2) #欧式距离  锚点与负采样，距离应该最大，与context，距离最近
                    #这2步有什么作用
                    #print(len(fips))
                    #index=np.arange(fips.size(0))#size(0)返回该二维矩阵的行数
                    index = np.arange(len(fips))
                    np.random.shuffle(index)  #在一个batch中打乱循序，得到远离的图片
                    distance2=dist(outputs1,outputs2[index]) #其他随机的
                    del(index)
                    #labels=torch.ones(fips.size(0)) #labels 是什么？
                    labels=torch.ones(len(fips)) #labels 是什么？
                    labels=labels.to(device)
                    loss= criterion(distance2, distance1, target=labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    del(anc_images)
                    del(pos_images)
                    del(outputs1)
                    del(outputs2)
                    del(labels)
                # statistics
                #running_loss += loss.item() * fips.size(0)
                running_loss += loss.item() * len(fips)
                stats['T'] += torch.sum(distance1 < distance2).cpu().item() #邻域内的小于随机的，所以是ture
                stats['F'] += torch.sum(distance1 > distance2).cpu().item()
                print(stats['F'])
                del(distance1) #删除变量？
                del(distance2)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_metric_value = metrics(stats)
            #embs.append(outputs1.detach().cpu().numpy()) #但是有相同小区的embedding ，取平均

            if verbose:
                print('{} Loss: {:.4f} Metrics: {:.4f}'.format(phase, epoch_loss, epoch_metric_value))

            training_log['current_epoch'] = epoch
            if phase == 'val':
                training_log['val_metric_value_history'].append(epoch_metric_value)
                training_log['val_loss_history'].append(epoch_loss)
                # deep copy the model
                if epoch_metric_value > best_metric_value:
                    best_metric_value = epoch_metric_value
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_optimizer_wts = copy.deepcopy(optimizer.state_dict())
                    best_log = copy.deepcopy(training_log)
                    nodecrease = 0
                else:
                    nodecrease += 1
            else:  # train phase
                training_log['train_loss_history'].append(epoch_loss)
                if scheduler != None:
                    scheduler.step()

            if nodecrease >= early_stop_epochs:
                early_stop = True

        if save_dir and epoch % save_epochs == 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_log': training_log
            }
            torch.save(checkpoint,
                       os.path.join(save_dir, model_name + '_' + str(training_log['current_epoch']) + '.tar'))

        if if_early_stop and early_stop:
            print('Early stopped!')
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best validation metric value: {:4f}'.format(best_metric_value))

    # load best model weights
    if return_best:
        model.load_state_dict(best_model_wts)
        optimizer.load_state_dict(best_optimizer_wts)
        training_log = best_log

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_log': training_log
    }

    if save_dir:
        torch.save(checkpoint,
                   os.path.join(save_dir, model_name + '_' + str(training_log['current_epoch']) + '_last.tar'))    
    return model, training_log, best_metric_value, embs


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]),
}

if __name__ == '__main__':
    with open(train_path_list_path, 'rb') as f:
        train_path_list = pickle.load(f)[0:64]
       # train_path_list = train_path_list[0:64]
    with open(val_path_list_path, 'rb') as f:
        val_path_list = pickle.load(f)[0:32]
       # val_path_list = val_path_list[0:32]
    print('training set size: ' + str(len(train_path_list)))
    print('validation set size: ' + str(len(val_path_list)))

    #PlaceImagePairDataset ？？
    datasets = {'train': PlaceImagePairDataset(data_dir, train_path_list, data_transforms['train']),
                'val': PlaceImagePairDataset(data_dir, val_path_list, data_transforms['val'])}

    dataloaders_dict = {x: DataLoader(datasets[x], batch_size=batch_size,
                                      shuffle=False, num_workers=0) for x in ['train', 'val']}
    best_metric=0
    best_lr=-1
    best_mr=-1
    #learning_rate = [0.0005]
    for i in learning_rate:
        for j in margin: 
            #margin =[5]
            #model
            model = PlaceImageSkipGram(embedding_dim=embedding_dim)
            media_dir=os.path.join(ckpt_save_dir,"lr%f_mr%f"%(i,j))
            pdist=torch.nn.PairwiseDistance(p=2.0)
            if not os.path.exists(media_dir):
                os.makedirs(media_dir)
            if not trainable_params == None:
                model.only_train(trainable_params=trainable_params)  # only train the selected parameters or modules
            model = model.to(device)
            #优化器
            optimizer = optim.Adam(model.parameters(), lr=i, betas=(0.9, 0.999), eps=1e-08,
                                   weight_decay=weight_decay, amsgrad=True)

            #loss
            loss_fn = torch.nn.MarginRankingLoss(margin=j)  #margin = [5]
            #功能： 计算两个向量之间的相似度，当两个向量之间的距离大于margin，则loss为正，小于margin，loss为0
            #等间隔调整学习率，调整倍数为gamma倍
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_epochs, gamma=lr_decay_rate)
            #训练
            _, training_log,best_value = train_embedding(model, model_name=model_name, dataloaders=dataloaders_dict, criterion=loss_fn,
                                   optimizer=optimizer, metrics=metrics, num_epochs=num_epochs, threshold=threshold,
                                   verbose=True, return_best=return_best,
                                   if_early_stop=if_early_stop, early_stop_epochs=early_stop_epochs, scheduler=scheduler,
                                   save_dir=media_dir, save_epochs=save_epochs,dist=pdist)
            media_dir1=os.path.join(media_dir,"training_log.txt")
            with open(media_dir1,"w") as file:
                for k in range(len(training_log["val_metric_value_history"])):
                    file.write("epoch:"+str(k)+"\n")
                    file.write("val_metric_value_history:"+str(training_log["val_metric_value_history"][k])+"\n")
                    file.write("val_loss_history:"+str(training_log["val_loss_history"][k])+"\n")
                    file.write("train_loss_history:"+str(training_log["train_loss_history"][k])+"\n")
            if best_value>best_metric:
                best_metric=best_value
                best_lr=i
                best_mr=j
    print("best_lr:"+str(best_lr)+" best_mr:"+str(best_mr)+" best_metric_value:"+str(best_metric))
                
