# pytorch_utils

### Description

pytorch框架下各种补充功能的实现，弥补一些功能的不足和缺失

### Usage

将utils文件夹整个拷贝到自己的工程文件下，将其当成一个包import使用就行

### Files Structure & Usage

#### averagemeter.py 
  
Desc：用于求均值，常用于求多个batch的loss均值

Usage:
```python
avg_loss = AverageMeter()
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
        data, target = data.to(device), target.to(device)
        out = model(data)
        loss = F.cross_entropy(out, target)
        
        avg_loss.update(loss)

        pred = out.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()

print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
    avg_loss.avg, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
```

#### loggers.py

Desc: 用于把终端的输出打印到文件中

Usage:
```python
logger_name = time.strftime('%Y-%m-%d-%H-%M-%S-') + 'train.log'
sys.stdout = Logger(os.path.join('log', DLmodel.model_name, logger_name))   # just for example
```

#### lr_scheduler.py

- WarmupWithCosineDecay
    
    先Warmup，后余弦衰减从最大lr衰减到最小lr的学习率更新策略。用法与PyTorch的API一致。

#### loss_func.py

- CrossEntropyLossWithLabelSmooth

  带LabelSmooth的CrossEntropy Loss

#### dataset_utils.py

Desc: 计算与多分类数据集相关的参数，如normalize所需的mean和std、用于nll loss加权用的各类类别权重计算。

- get_mean_and_std

  功能：计算数据集中图像的均值和标准差，用于normalize
  
  接收参数：torch.dataset类型或继承自该类型的表示数据集的变量
  
- get_class_weight

  功能：计算数据集中每类的加权权重，计算公式：1/n*N，其中n是当前类的样本数量，N是数据集中所有样本的数量

  接收参数：torch.dataset类型或继承自该类型的表示数据集的变量
  



### TODO List
- [ ] Augmix图像增强
- [X] 添加labelsmoothing的loss
- [X] 测试lr_scheduler.py中的WarmupWithCosineDecay
