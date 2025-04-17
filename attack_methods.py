import torch
import scipy.stats as st
import numpy as np
import torchvision
from torchvision.models import resnet18, resnext50_32x4d
from PIL import Image
import random
import time
import math
import torch.nn.functional as F
from pick_restore import mask_video_frames, restore_masked_frames
import cv2
import torchvision.models as models
import torch.nn as nn

def norm_grads(grads, frame_level=True):
    # frame level norm
    # clip level norm
    assert len(grads.shape) == 5 and grads.shape[2] == 32
    if frame_level:
        norm = torch.mean(torch.abs(grads), [1,3,4], keepdim=True)
    else:
        norm = torch.mean(torch.abs(grads), [1,2,3,4], keepdim=True)
    # norm = torch.norm(grads, dim=[1,2,3,4], p=1)
    return grads / norm

class Attack(object):
    """
    # refer to https://github.com/Harry24k/adversarial-attacks-pytorch
    Base class for all attacks.
    .. note::
        It automatically set device to the device where given model is.
        It temporarily changes the model's training mode to `test`
        by `.eval()` only during an attack process.
    """
    def __init__(self, name, model):
        r"""
        Initializes internal attack state.
        Arguments:
            name (str) : name of an attack.
            model (torch.nn.Module): model to attack.
        """
        self.attack = name
        self.model = model
        self.model_name = str(model).split("(")[0]

        self.training = model.training
        self.device = next(model.parameters()).device
        
        self._targeted = 1
        self._attack_mode = 'default'
        self._return_type = 'float'
        self._target_map_function = lambda images, labels:labels

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def forward(self, *input):
        r"""
        It defines the computation performed at every call (attack forward).
        Should be overridden by all subclasses.
        """
        raise NotImplementedError
        
    def set_attack_mode(self, mode, target_map_function=None):
        r"""
        Set the attack mode.
  
        Arguments:
            mode (str) : 'default' (DEFAULT)
                         'targeted' - Use input labels as targeted labels.
                         'least_likely' - Use least likely labels as targeted labels.
                         
            target_map_function (function) :
        """
        if self._attack_mode == 'only_default':
            raise ValueError("Changing attack mode is not supported in this attack method.")
            
        if (mode == 'targeted') and (target_map_function is None):
            raise ValueError("Please give a target_map_function, e.g., lambda images, labels:(labels+1)%10.")
            
        if mode=="default":
            self._attack_mode = "default"
            self._targeted = 1
            self._transform_label = self._get_label
        elif mode=="targeted":
            self._attack_mode = "targeted"
            self._targeted = -1
            self._target_map_function = target_map_function
            self._transform_label = self._get_target_label
        elif mode=="least_likely":
            self._attack_mode = "least_likely"
            self._targeted = -1
            self._transform_label = self._get_least_likely_label
        else:
            raise ValueError(mode + " is not a valid mode. [Options : default, targeted, least_likely]")
            
    def set_return_type(self, type):
        r"""
        Set the return type of adversarial images: `int` or `float`.
        Arguments:
            type (str) : 'float' or 'int'. (DEFAULT : 'float')
        """
        if type == 'float':
            self._return_type = 'float'
        elif type == 'int':
            self._return_type = 'int'
        else:
            raise ValueError(type + " is not a valid type. [Options : float, int]")

    def save(self, save_path, data_loader, verbose=True):
        r"""
        Save adversarial images as torch.tensor from given torch.utils.data.DataLoader.
        Arguments:
            save_path (str) : save_path.
            data_loader (torch.utils.data.DataLoader) : data loader.
            verbose (bool) : True for displaying detailed information. (DEFAULT : True)
        """
        self.model.eval()

        image_list = []
        label_list = []

        correct = 0
        total = 0

        total_batch = len(data_loader)

        for step, (images, labels) in enumerate(data_loader):
            adv_images = self.__call__(images, labels)

            image_list.append(adv_images.cpu())
            label_list.append(labels.cpu())

            if self._return_type == 'int':
                adv_images = adv_images.float()/255

            if verbose:
                outputs = self.model(adv_images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(self.device)).sum()

                acc = 100 * float(correct) / total
                print('- Save Progress : %2.2f %% / Accuracy : %2.2f %%' % ((step+1)/total_batch*100, acc), end='\r')

        x = torch.cat(image_list, 0)
        y = torch.cat(label_list, 0)
        torch.save((x, y), save_path)
        print('\n- Save Complete!')

        self._switch_model()
    
    def _transform_video(self, video, mode='forward'):
        r'''
        Transform the video into [0, 1]
        '''
        dtype = video.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=self.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=self.device)
        if mode == 'forward':
            # [-mean/std, mean/std]
            video.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
        elif mode == 'back':
            # [0, 1]
            video.mul_(std[:, None, None, None]).add_(mean[:, None, None, None])
        return video

    def _transform_label(self, images, labels):
        r"""
        Function for changing the attack mode.
        """
        return labels
        
    def _get_label(self, images, labels):
        r"""
        Function for changing the attack mode.
        Return input labels.
        """
        return labels
    
    def _get_target_label(self, images, labels):
        r"""
        Function for changing the attack mode.
        Return input labels.
        """
        return self._target_map_function(images, labels)
    
    def _get_least_likely_label(self, images, labels):
        r"""
        Function for changing the attack mode.
        Return least likely labels.
        """
        outputs = self.model(images)
        _, labels = torch.min(outputs.data, 1)
        labels = labels.detach_()
        return labels
    
    def _to_uint(self, images):
        r"""
        Function for changing the return type.
        Return images as int.
        """
        return (images*255).type(torch.uint8)

    def _switch_model(self):
        r"""
        Function for changing the training mode of the model.
        """
        if self.training:
            self.model.train()
        else:
            self.model.eval()

    def __str__(self):
        info = self.__dict__.copy()
        
        del_keys = ['model', 'attack']
        
        for key in info.keys():
            if key[0] == "_" :
                del_keys.append(key)
                
        for key in del_keys:
            del info[key]
        
        info['attack_mode'] = self._attack_mode
        if info['attack_mode'] == 'only_default' :
            info['attack_mode'] = 'default'
            
        info['return_type'] = self._return_type
        
        return self.attack + "(" + ', '.join('{}={}'.format(key, val) for key, val in info.items()) + ")"

    def __call__(self, *input, **kwargs):
        self.model.eval()
        images = self.forward(*input, **kwargs)
        self._switch_model()

        if self._return_type == 'int':
            images = self._to_uint(images)

        return images

class STKF(Attack):

    def __init__(self, model, params,mask_num,top_k,epsilon=16/255, steps=1, delay=1.0):
        super(STKF, self).__init__("STKF", model)

        num_classes = 400
        hidden_size = 256
        num_layers = 1
        self.cnn = models.resnet18(pretrained=True).cuda()
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])  # 移除全连接层
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True).cuda()
        self.fc1 = nn.Linear(hidden_size, num_classes).cuda()

        # 视觉重要性特征提取
        self.backbone = models.resnet18(pretrained=True).cuda()
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        self.fc = nn.Linear(512, 1).cuda()  # 输出单通道重要性分数

        self.epsilon = 0.03
        self.alpha = 0.01
        self.iterations = 10


    def VideoLSTM(self, x):
        batch_size, timesteps, C, H, W = x.size()
        # 合并批次和时间维度
        c_in = x.view(batch_size * timesteps, C, H, W).to(x.device)
        c_out = self.cnn(c_in).squeeze()  # (batch*timesteps, 512)
        # 恢复时序维度
        r_in = c_out.view(batch_size, timesteps, -1)
        # LSTM处理
        r_out, _ = self.lstm(r_in)
        # 取最后一个时间步输出
        output = self.fc1(r_out[:, -1, :])
        return output

    # 模块1：视觉重要性特征提取
    def importance_model(self, x):
        # 输入形状：(B, T, C, H, W)
        B, T = x.shape[:2]
        x = x.view(B * T, *x.shape[2:])
        features = self.feature_extractor(x).squeeze()
        scores = self.fc(features).view(B, T)
        return torch.sigmoid(scores)  # 归一化到[0,1]

    # 模块2：光流能量计算（使用OpenCV Farneback算法）
    def compute_optical_energy(self, frames):
        """
        输入：frames tensor (T, H, W, C)
        输出：光流能量 tensor (T-1,)
        """
        energies = []
        frames = frames.cpu().numpy().astype(np.uint8)
        for i in range(len(frames) - 1):
            try:
                prev = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            except Exception as e:
                print(f"Error converting color: {e}")
            next_ = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)

            # Farneback参数设置
            flow = cv2.calcOpticalFlowFarneback(
                prev, next_, None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )

            # 计算L2范数能量
            energy = np.sum(flow ** 2)
            energies.append(energy)

        # 对齐时间维度（最后补零）
        energies.append(0)
        return torch.FloatTensor(energies)

    def AdversarialGenerator(self, model, x, optical_energy):
        """
                x: 原始视频张量 (B, T, C, H, W)
                optical_energy: 光流能量 (B, T)
                """
        x_adv = x.clone().detach().requires_grad_(True)
        for _ in range(self.iterations):
            # 计算视觉重要性
            vis_scores = self.importance_model(x_adv)  # (B, T)

            # 能量标准化
            flow_scores = optical_energy / optical_energy.max()

            # 融合得分
            combined_scores = 0.7 * vis_scores + 0.3 * flow_scores  # 可调权重

            # 应用重要性加权的FGSM攻击
            perturbation = self.alpha * x_adv.sign()
            perturbation *= combined_scores.view(*combined_scores.shape, 1, 1, 1)  # 维度对齐

            # 更新对抗样本
            x_adv = x_adv.detach() + perturbation
            x_adv = torch.clamp(x_adv, x - self.epsilon, x + self.epsilon)
            x_adv = torch.clamp(x_adv, 0, 1)


        return x_adv

    def forward(self, videos, labels):

        videos = videos.permute(0, 2, 1, 3, 4)  # (N, T, C, H, W)

        start_time = time.time()
        # 计算光流能量（需在CPU处理）
        optical_energies = []
        for video in videos:
            energy = self.compute_optical_energy(video.permute(0, 2, 3, 1))  # (T, H, W, C)
            optical_energies.append(energy)
        optical_energy = torch.stack(optical_energies).to('cuda')

        # 生成对抗样本
        adv_video = self.AdversarialGenerator(
            model=self.VideoLSTM,
            x=videos.cuda(),
            optical_energy=optical_energy
        )

        print('now_time', time.time() - start_time)
        print("对抗样本形状:", adv_video.shape)
        adv_video = adv_video.permute(0, 2, 1, 3, 4)
        return adv_video

