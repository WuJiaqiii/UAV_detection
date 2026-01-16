import os
import torch
import logging
import numpy as np
import random
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
import seaborn as sns
import torch.distributed as dist

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def _reduce_scalar(val, device, dtype=torch.float32, op=dist.ReduceOp.SUM):
    if isinstance(val, torch.Tensor):
        t = val.detach()
        if dtype is not None and t.dtype != dtype:
            t = t.to(dtype)
        if device is not None and t.device != torch.device(device):
            t = t.to(device)
        t = t.clone()   
    else:
        t = torch.scalar_tensor(float(val), dtype=dtype, device=device)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=op)
        if op == dist.ReduceOp.SUM:
            ws = dist.get_world_size()
            if ws > 1:
                t /= ws

    return t.item()

def _set_epoch_for_loaders(epoch, *loaders):
    for ld in loaders:
        if ld is not None and hasattr(ld, "sampler") and hasattr(ld.sampler, "set_epoch"):
            ld.sampler.set_epoch(epoch)
            
class ColorFormatter(logging.Formatter):
    # ANSI 颜色表
    COLORS = {
        logging.DEBUG:    "\033[36m",        # 青色
        logging.INFO:     "\033[32m",        # 绿色
        logging.WARNING:  "\033[33m",        # 黄色
        logging.ERROR:    "\033[31m",        # 红色
        logging.CRITICAL: "\033[1;41m",      # 白字红底
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        record.msg = f"{color}{record.msg}{self.RESET}"
        return super().format(record)

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
        
class ExpLogger(logging.Logger):
    def init_exp(self, config):
        super().info('---------------------Experiment Settings-------------------------')
        super().info('Nothing here')
        super().info('-----------------------------------------------------------------')
        
logging.setLoggerClass(ExpLogger)

def create_logger(filename: str) -> logging.Logger:
    
    def custom_time(*args):
        utc_plus_8 = datetime.now(tz=timezone.utc) + timedelta(hours=8)
        return utc_plus_8.timetuple()
        
    logger = logging.getLogger(filename)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    formatter.converter = custom_time
    ch.setFormatter(formatter)
    
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger

class EarlyStopping:
    def __init__(self, logger, patience=10, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.logger = logger

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.logger.info(
                f'--Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.logger.info(f'--EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.logger.info(
                f'--Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            self.val_loss_min = val_loss
            self.counter = 0

def compute_cluster_accuracy(real_labels, pred_labels):
    real_labels = torch.tensor(real_labels)
    pred_labels = torch.tensor(pred_labels)
    unique_real = torch.unique(real_labels)
    unique_pred = torch.unique(pred_labels)
    
    cost_matrix = torch.zeros((len(unique_real), len(unique_pred)), dtype=torch.int32)
    
    for i, real_label in enumerate(unique_real):
        for j, pred_label in enumerate(unique_pred):
            matches = torch.sum((real_labels == real_label) & (pred_labels == pred_label))
            cost_matrix[i, j] = -matches

    row_ind, col_ind = linear_sum_assignment(cost_matrix.numpy())
    matched_labels = pred_labels.clone()
    for i, j in zip(row_ind, col_ind):
        matched_labels[pred_labels == unique_pred[j]] = unique_real[i]

    return accuracy_score(real_labels.numpy(), matched_labels.numpy())
    
def save_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()