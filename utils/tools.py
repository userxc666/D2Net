import math

import numpy as np
import torch
import matplotlib.pyplot as plt
import time

plt.switch_backend('agg')

def sgdr_lr(epoch, args):
    """SGDR with decaying max_lr"""
    total_epochs = args.train_epochs
    base_lr      = args.learning_rate
    warmup_frac  = 0.1        # 前 10% 轮线性 warm-up
    cycles       = 3          # 后续分成多少个 restart cycle

    # 1) Warm-up 部分
    warmup_epochs = int(warmup_frac * total_epochs)
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs

    # 2) SGDR 部分
    cur = epoch - warmup_epochs
    rem = total_epochs - warmup_epochs
    cycle_len = rem // cycles
    cycle_idx = min(cur // cycle_len, cycles - 1)
    cycle_epoch = cur % cycle_len

    # 每次 restart 后 peak_lr 衰减一半
    peak_lr = base_lr * (0.5 ** cycle_idx)
    progress = cycle_epoch / cycle_len
    return 0.5 * peak_lr * (1 + math.cos(math.pi * progress))

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    
    # Sigmoid learning rate decay
    elif args.lradj == 'sigmoid':
        if args.data_path == 'traffic.csv':
            if args.pred_len == 96:
                k,s,w = 0.5,10,10
            elif args.pred_len == 192:
                k,s,w = 0.3,8,5
            elif args.pred_len == 336:
                k,s,w = 0.25,10,6
            else:
                k,s,w = 0.2,12,8
        elif args.data_path == 'electricity.csv':
            k,s,w = 0.3,5,20
        else:
           k,s,w = 0.5,10,10
            # k = 0.3# logistic growth rate  national_illness.csv
            # s = 6  # decreasing curve smoothing rate
            # w = 12# warm-up coefficient
        lr_adjust = {epoch: args.learning_rate / (1 + np.exp(-k * (epoch - w))) - args.learning_rate / (1 + np.exp(-k/s * (epoch - w*s)))}

    elif args.lradj == 'sgdr':
        lr = sgdr_lr(epoch, args)  # 如果 epoch 从 1 开始则 epoch-1，否则直接 epoch
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
        return

    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
 
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')