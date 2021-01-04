import torch
from tqdm import tqdm

def get_mean_and_std(dataset):
    '''
    :desc Compute the mean and std value of dataset.
    :param dataset: torch.dataset class
    '''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in tqdm(dataloader):
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def get_class_weight(dataset):
    """
    :desc 计算每类的权重，用于NLLLoss给各类加权
    :param dataset: torch.dataset class
    """
    class_weight = [0] * len(dataset.classes)
    for _, class_idx in tqdm(dataset):
        class_weight[class_idx] += 1
    for idx, val in enumerate(class_weight):
        class_weight[idx] = 1/val*len(dataset)/len(dataset.classes)
    return class_weight