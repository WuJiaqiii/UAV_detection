import torch
import logging
import numpy as np
import random
from datetime import datetime, timedelta, timezone
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