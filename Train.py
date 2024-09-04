#!/usr/bin/env python

# @Time    : 2024/3/28 21:39
# @Author  : Yao Xuan
# @Email   : xuany0512@163.com
# @File    : Train.py
import pHLAformer
import os.path
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import Dataset
import Utils
import math
import argparse
import random


parser = argparse.ArgumentParser(description='train the pHLAformer model')
parser.add_argument('--train_file', type=str, help='MHC-pep pairs csv with columns ["allele", "peptide", "label"]')
parser.add_argument('--val_file', type=str, help='MHC-pep pairs csv with columns ["allele", "peptide", "label"]')
parser.add_argument('--save_dir', type=str, default="./output/", help='output dir of model')

# Hyperparameters
parser.add_argument('--batch_size', type=int, default=512, help='mini batchsize')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
parser.add_argument('--max_epoch', type=int, default=50, help='the max training epoch number')
parser.add_argument('--gamma', type=int, default=0.6, help='Multiplicative factor of learning rate decay.')
args = parser.parse_args()

# 设置随机初始化种子
seed = 20000115


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(seed)



def train():
    train_file = args.train_file
    val_file = args.val_file

    save_dir = args.save_dir
    if not os.path.exists((save_dir)):
        os.makedirs(save_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predictor = pHLAformer.pHLAformer()
    gamma = 0.6
    train_batch_size = 512
    init_lr = 0.0001
    epochs = 10

    train_data = Dataset.HLAPEPDataset(file_path=train_file)
    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, drop_last=False)
    step_size = math.floor(train_data.__len__() / train_batch_size)
    val_data = Dataset.HLAPEPDataset(file_path=val_file)
    val_loader = DataLoader(val_data, batch_size=train_batch_size, shuffle=True,drop_last=False)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(predictor.parameters(),lr=init_lr)
    StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    train_loss_dic = {}
    val_loss_dic = {}

    all_train_losses = []
    all_val_losses = []

    epoch_train_losses = []
    epoch_val_losses = []

    epoch_val_pcc = []
    epoch_test_pcc = []

    val_pcc_dic = {}

    for epoch in range(epochs):
        predictor.train()
        train_losses = []
        val_losses = []
        for step, (hla_seq, pep_seq, label) in enumerate(train_loader):
            score , _ = predictor(pep_seq = pep_seq, hla_seq = hla_seq)
            label = label.unsqueeze(dim=1)
            optimizer.zero_grad()
            train_loss = criterion(score.float(), label.to(device).float())
            train_loss.backward()
            optimizer.step()
            StepLR.step()
            print(optimizer.param_groups[0]['lr'])
            all_train_losses.append(train_loss.item())
            train_losses.append(train_loss.item())
            print('train-epoch:{},batch:{},loss:{}'.format(epoch, step, train_loss.data))

        epoch_train_losses.append(sum(train_losses)/len(train_losses))
        train_loss_dic['epoch:' + str(epoch)] = train_losses


        # evaluation
        predictor.eval()
        with torch.no_grad():
            val_label = []
            val_preds = []
            for step, (hla_seq, pep_seq, label) in enumerate(val_loader):
                score , _ = predictor(pep_seq = pep_seq,hla_seq = hla_seq)
                label = label.unsqueeze(dim=1)
                val_loss = criterion(score.float(),label.to(device).float())
                print('val-epoch:{},batch:{},loss:{}'.format(epoch, step, val_loss.data))
                val_losses.append(val_loss.item())
                all_val_losses.append(val_loss.item())
                val_label += label.squeeze().tolist()
                val_preds += score.squeeze().tolist()
            epoch_val_losses.append(sum(val_losses)/len(val_losses))
            val_loss_dic['epoch:'+str(epoch)] = val_losses

            val_df = pd.DataFrame()
            val_df['labels'] = val_label
            val_df['preds'] = val_preds
            val_pcc = np.corrcoef(val_df['preds'], val_df['labels'])[0][1]
            epoch_val_pcc.append(val_pcc)
            val_pcc_dic['epoch:' + str(epoch)] = [val_pcc]


    train_loss_df = pd.DataFrame(train_loss_dic)
    val_loss_df = pd.DataFrame(val_loss_dic)
    val_pcc_df = pd.DataFrame(val_pcc_dic)

    Utils.plot_pcc(data=epoch_val_pcc,color='pink',label='Validation Performance',file_name='val_pcc_plot')
    Utils.plot_pcc(data=epoch_test_pcc, color='orange', label='Test Performance', file_name='test_pcc_plot')

    # save
    train_loss_df.to_csv(save_dir + 'onehot_el_v1_train_loss.csv', index=False)
    val_loss_df.to_csv(save_dir + 'onehot_el_v1_val_loss.csv', index=False)
    val_pcc_df.to_csv(save_dir + 'ESM2_affinity_v1_val_pcc.csv', index=False)
    torch.save(predictor.state_dict(), save_dir + 'model.pt')

