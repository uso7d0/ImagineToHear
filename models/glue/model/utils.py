import math
import time

import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr 

def calc_f1_acc(pred, label, return_sum=False):
    _, idx = pred.max(1)
    x = label.cpu().numpy()
    y = idx.cpu().numpy()
    acc = accuracy_score(x, y)
    if return_sum:
        return acc * len(x)
    return acc

def calc_spearmanr(pred, label):
    pred_vals = pred.squeeze(-1).cpu().numpy()
    label_vals = label.cpu().numpy()
    return spearmanr(pred_vals, label_vals).correlation

class AverageMeter(object):
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

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (remain %s)" % (asMinutes(s), asMinutes(rs))
