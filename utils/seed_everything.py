import random
import os
import torch
import numpy as np

def seed_everything(seed):
    """
    :description:   set the random seed to control randomness in order to reproduce the result,
                    you can run it at the start of your program
    :param seed:    random seed, int value
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True