import torch
import numpy as np

class WarmupWithCosine(object):
    def __init__(self, optimizer, lr_max=0.01, lr_min=0, warm_milestone=10, total_milestone=100):
        """
        Desc:           先warmup，后余弦decay的学习率调整策略，大致遵循了pytorch的API的一贯写法
        :optimizer：     优化器
        :lr_max：        warmup结束后达到的最高lr
        :lr_min:        余弦降低后达到的最低lr_min，即最后一个epoch时的lr
        :warm_milestone:warmup总共需要的epoch次数
        :total_milestone:总共迭代次数，用来计算Cosine decay的
        """
        self.optimizer = optimizer
        self.epoch_cnt = 0
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.warm_milestone = warm_milestone
        self.total_milestone = total_milestone

    def step(self, epoch=None):
        if epoch:
            self.epoch_cnt = epoch

        if self.epoch_cnt <= self.warm_milestone:
            lr_val = self.lr_max / self.warm_milestone * self.epoch_cnt

        else:
            lr_val = (np.cos(2 * np.pi * (self.epoch_cnt - self.warm_milestone) / (
                    2 * (self.total_milestone - self.warm_milestone))) + 1) / 2 * self.lr_max
        self.epoch_cnt += 1
        return self.__set_lr(lr_val)

    def __set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return self.get_lr() == lr  # check whether to set correctly

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']