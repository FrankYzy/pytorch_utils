import torch
import numpy as np

######################## MIX UP ###########################
"""
Desc：mixup图像增强/模型复杂度控制技巧
Ref：代码抽取自https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py
Explain：将一个大小为n的batch中的多个图像随机两两匹配，用beta随机分布生成权重给二者加权，共生成n个mixup的新样本
Usage:
    >>> from utils.data_augmentation import *
    >>> from utils.loss_func import *                       # optional
    >>> criterion = CrossEntropyLossWithLabelSmooth(0.1)    # or any other loss function is ok
    >>> ...
    >>> for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            inputs, targets_a, targets_b, lam = mixup_data(data, target)
            out = model(inputs)
            loss = mixup_criterion(criterion, out, targets_a, targets_b, lam)
"""
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    """
    :param alpha: any alpha > 0 has the same effect. 
                    if not alpha > 0: same to diable the mixup function
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)