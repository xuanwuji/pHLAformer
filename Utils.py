#!/usr/bin/env python

# @Time    : 2024/9/4 15:33
# @Author  : Yao Xuan
# @Email   : xuany0512@163.com
# @File    : Utils.py
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_loss(data, color, label, file_name):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.lineplot(x=range(epochs), y=data, color=color, label=label)
    ax.set_xlabel('Epoch', fontsize=16)
    ax.set_ylabel('Loss', fontsize=16)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(save_dir + file_name + '.jpg', dpi=300)
    plt.savefig(save_dir + file_name + '.svg', format='svg', dpi=300)
    ax.legend(fontsize=14)


plot_loss(data=epoch_train_losses, label='Train Loss', color='red', file_name='train_loss_plot')
plot_loss(data=epoch_val_losses, label='Validation Loss', color='green', file_name='val_loss_plot')
plot_loss(data=epoch_test_losses, label='Test Loss', color='grey', file_name='test_loss_plot')


def plot_pcc(data, color, label, file_name):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.lineplot(x=range(epochs), y=data, color=color, label=label)
    ax.set_xlabel('Epoch', fontsize=16)
    ax.set_ylabel('Pearson Correlation Coefficient', fontsize=16)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(save_dir + file_name + '.jpg', dpi=300)
    plt.savefig(save_dir + file_name + '.svg', format='svg', dpi=300)
    ax.legend(fontsize=14)