#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-26
import os
from collections import Sequence

import cv2
import numpy as np
import torch
import matplotlib.cm as cm
import torch.nn as nn
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms
from tqdm import tqdm

########################## API的底层函数 #########################
class _BaseWrapper(object):
    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image):
        self.image_shape = image.shape[2:]
        self.logits = self.model(image)
        self.probs = F.softmax(self.logits, dim=1)
        return self.probs.sort(dim=1, descending=True)  # ordered results

    def backward(self, ids):
        """
        Class-specific backpropagation
        """
        one_hot = self._encode_one_hot(ids)
        self.model.zero_grad()
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()


class BackPropagation(_BaseWrapper):
    def forward(self, image):
        self.image = image.requires_grad_()
        return super(BackPropagation, self).forward(self.image)

    def generate(self):
        gradient = self.image.grad.clone()
        self.image.grad.zero_()
        return gradient


class GuidedBackPropagation(BackPropagation):
    """
    "Striving for Simplicity: the All Convolutional Net"
    https://arxiv.org/pdf/1412.6806.pdf
    Look at Figure 1 on page 8.
    """

    def __init__(self, model):
        super(GuidedBackPropagation, self).__init__(model)

        def backward_hook(module, grad_in, grad_out):
            # Cut off negative gradients
            if isinstance(module, nn.ReLU):
                return (F.relu(grad_in[0]),)

        for module in self.model.named_modules():
            self.handlers.append(module[1].register_backward_hook(backward_hook))


class Deconvnet(BackPropagation):
    """
    "Striving for Simplicity: the All Convolutional Net"
    https://arxiv.org/pdf/1412.6806.pdf
    Look at Figure 1 on page 8.
    """

    def __init__(self, model):
        super(Deconvnet, self).__init__(model)

        def backward_hook(module, grad_in, grad_out):
            # Cut off negative gradients and ignore ReLU
            if isinstance(module, nn.ReLU):
                return (F.relu(grad_out[0]),)

        for module in self.model.named_modules():
            self.handlers.append(module[1].register_backward_hook(backward_hook))


class GradCAM(_BaseWrapper):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """

    def __init__(self, model, candidate_layers=None):
        super(GradCAM, self).__init__(model)
        self.fmap_pool = {}
        self.grad_pool = {}
        self.candidate_layers = candidate_layers  # list

        def save_fmaps(key):
            def forward_hook(module, input, output):
                self.fmap_pool[key] = output.detach()

            return forward_hook

        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook

        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = F.adaptive_avg_pool2d(grads, 1)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)
        gcam = F.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )

        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam


def occlusion_sensitivity(
    model, images, ids, mean=None, patch=35, stride=1, n_batches=128
):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure A5 on page 17

    Originally proposed in:
    "Visualizing and Understanding Convolutional Networks"
    https://arxiv.org/abs/1311.2901
    """

    torch.set_grad_enabled(False)
    model.eval()
    mean = mean if mean else 0
    patch_H, patch_W = patch if isinstance(patch, Sequence) else (patch, patch)
    pad_H, pad_W = patch_H // 2, patch_W // 2

    # Padded image
    images = F.pad(images, (pad_W, pad_W, pad_H, pad_H), value=mean)
    B, _, H, W = images.shape
    new_H = (H - patch_H) // stride + 1
    new_W = (W - patch_W) // stride + 1

    # Prepare sampling grids
    anchors = []
    grid_h = 0
    while grid_h <= H - patch_H:
        grid_w = 0
        while grid_w <= W - patch_W:
            grid_w += stride
            anchors.append((grid_h, grid_w))
        grid_h += stride

    # Baseline score without occlusion
    baseline = model(images).detach().gather(1, ids)

    # Compute per-pixel logits
    scoremaps = []
    for i in tqdm(range(0, len(anchors), n_batches), leave=False):
        batch_images = []
        batch_ids = []
        for grid_h, grid_w in anchors[i : i + n_batches]:
            images_ = images.clone()
            images_[..., grid_h : grid_h + patch_H, grid_w : grid_w + patch_W] = mean
            batch_images.append(images_)
            batch_ids.append(ids)
        batch_images = torch.cat(batch_images, dim=0)
        batch_ids = torch.cat(batch_ids, dim=0)
        scores = model(batch_images).detach().gather(1, batch_ids)
        scoremaps += list(torch.split(scores, B))

    diffmaps = torch.cat(scoremaps, dim=1) - baseline
    diffmaps = diffmaps.view(B, new_H, new_W)

    return diffmaps

def save_gradient(filename, gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    #cv2.imwrite(filename, np.uint8(gradient))
    cv2.imencode('.jpg', np.uint8(gradient))[1].tofile(filename)

def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    #cv2.imwrite(filename, np.uint8(gcam))
    cv2.imencode('.jpg',np.uint8(gcam))[1].tofile(filename)


#################### 功能函数 #########################
def model_response_visualization(image_path, transform, model, target_layer, arch, topk, output_dir, classes, device):
    """
    :desc   利用GradCAM方法对指定层的响应进行可视化，生成attention热力图，解释模型学习性。
    :param  image_path: 要可视化热力图的图像文件的目录
            transform:  对该图像应用的transform预处理
            model:      加载好权重了的模型对象
            target_layer:要显示响应的模型层的名称（可以通过print(mode)获得模型中层的名字）
            arch:       无关紧要，一般传入模型就行
            topk:       要显示对当前图像预测的topk个类别分别对应的响应热力图
            output_dir: 生成的响应热力图结果保存的位置
            classes:    预测输出对应的类别名称列表
            device:     运行设备,cpu/cuda
    :额外注释： 本代码中还有一些注释掉的行，取消注释的话可以获得其他方法生成的热力图，但可视化效果不太好，就暂时注释了。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = model.to(device)
    model.eval()

    images = []
    raw_images = []
    print("Image:", image_path)

    raw_image = cv2.imread(image_path)
    image = Image.fromarray(raw_image[..., ::-1])
    raw_image = cv2.resize(raw_image, (224,) * 2)

    image = transform(image)

    images.append(image)
    raw_images.append(raw_image)

    images = torch.stack(images).to(device)

    # =========================================================================
    # print("Vanilla Backpropagation:")
    #
    bp = BackPropagation(model=model)
    probs, ids = bp.forward(images)  # sorted
    #
    # for i in range(topk):
    #     bp.backward(ids=ids[:, [i]])
    #     gradients = bp.generate()
    #
    #     # Save results as image files
    #     for j in range(len(images)):
    #         print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))
    #
    #         save_gradient(
    #             filename=os.path.join(
    #                 output_dir,
    #                 "{}-{}-vanilla-{}.png".format(j, arch, classes[ids[j, i]]),
    #             ),
    #             gradient=gradients[j],
    #         )
    #
    # # Remove all the hook function in the "model"
    # bp.remove_hook()

    # =========================================================================
    # print("Deconvolution:")
    #
    # deconv = Deconvnet(model=model)
    # _ = deconv.forward(images)
    #
    # for i in range(topk):
    #     deconv.backward(ids=ids[:, [i]])
    #     gradients = deconv.generate()
    #
    #     for j in range(len(images)):
    #         print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))
    #
    #         save_gradient(
    #             filename=os.path.join(
    #                 output_dir,
    #                 "{}-{}-deconvnet-{}.png".format(j, arch, classes[ids[j, i]]),
    #             ),
    #             gradient=gradients[j],
    #         )
    #
    # deconv.remove_hook()

    # =========================================================================
    print("Grad-CAM/Guided Backpropagation/Guided Grad-CAM:")

    gcam = GradCAM(model=model)
    _ = gcam.forward(images)

    gbp = GuidedBackPropagation(model=model)
    _ = gbp.forward(images)

    for i in range(topk):
        # Guided Backpropagation
        gbp.backward(ids=ids[:, [i]])
        gradients = gbp.generate()

        # Grad-CAM
        gcam.backward(ids=ids[:, [i]])
        regions = gcam.generate(target_layer=target_layer)

        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            # Guided Backpropagation
            # save_gradient(
            #     filename=os.path.join(
            #         output_dir,
            #         "{}-{}-guided-{}.png".format(j, arch, classes[ids[j, i]]),
            #     ),
            #     gradient=gradients[j],
            # )

            # Grad-CAM
            save_gradcam(
                filename=os.path.join(
                    output_dir,
                    "{}-{}-gradcam-{}-{}.png".format(
                        j, arch, target_layer, classes[ids[j, i]]
                    ),
                ),
                gcam=regions[j, 0],
                raw_image=raw_images[j],
            )

            # Guided Grad-CAM
            # save_gradient(
            #     filename=os.path.join(
            #         output_dir,
            #         "{}-{}-guided_gradcam-{}-{}.png".format(
            #             j, arch, target_layer, classes[ids[j, i]]
            #         ),
            #     ),
            #     gradient=torch.mul(regions, gradients)[j],
            # )



