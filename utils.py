import os
from gluoncv.torch.engine.config import get_cfg_defaults
import torch

# config info
# refer to https://cv.gluon.ai/model_zoo/action_recognition.html
CONFIG_ROOT = './config'  # config paths
# CONFIG_ROOT = '/root/autodl-tmp/STKF/STKF_attack/config'  # config paths
KINETICES_CONFIG_PATHS = {
    'i3d_resnet50': os.path.join(CONFIG_ROOT, 'i3d_nl5_resnet50_v1_kinetics400.yaml'),
    'i3d_resnet101': os.path.join(CONFIG_ROOT, 'i3d_nl5_resnet101_v1_kinetics400.yaml'),
    'slowfast_resnet50': os.path.join(CONFIG_ROOT, 'slowfast_8x8_resnet50_kinetics400.yaml'),
    'slowfast_resnet101': os.path.join(CONFIG_ROOT, 'slowfast_8x8_resnet101_kinetics400.yaml'),
    'tpn_resnet50': os.path.join(CONFIG_ROOT, 'tpn_resnet50_f32s2_kinetics400.yaml'),
    'tpn_resnet101': os.path.join(CONFIG_ROOT, 'tpn_resnet101_f32s2_kinetics400.yaml')
    }

# save info
OPT_PATH = './STKFOpt-path'  # output path
# OPT_PATH = '/root/autodl-tmp/STKFOpt-path'
# ucf model infos
UCF_MODEL_ROOT = './Checkpoint'  # ckpt file path of UCF101
# UCF_MODEL_ROOT = '/root/autodl-tmp/STKF/STKF_attack/Checkpoint' # ckpt file path of UCF101
UCF_MODEL_TO_CKPTS = {
    'i3d_resnet50': os.path.join(UCF_MODEL_ROOT, 'i3d_resnet50.pth'),
    'i3d_resnet101': os.path.join(UCF_MODEL_ROOT, 'i3d_resnet101.pth'),
    'slowfast_resnet50': os.path.join(UCF_MODEL_ROOT, 'slowfast_resnet50.pth'),
    'slowfast_resnet101': os.path.join(UCF_MODEL_ROOT, 'slowfast_resnet101.pth'),
    'tpn_resnet50': os.path.join(UCF_MODEL_ROOT, 'tpn_resnet50.pth'),
    'tpn_resnet101': os.path.join(UCF_MODEL_ROOT, 'tpn_resnet101.pth')
}
# ucf dataset
UCF_DATA_ROOT = './Dataset-ucf101'   # ucf101 dataset path
Kinetic_DATA_ROOT = './Dataset-kinetics'  # kinetics dataset path
# UCF_DATA_ROOT = '/root/autodl-tmp/STKF/STKF_attack/Dataset-ucf101'   # ucf101 dataset path
# Kinetic_DATA_ROOT = '/root/autodl-tmp/STKF/STKF_attack/Dataset-kinetics'  # kinetics dataset path

def change_cfg(cfg, batch_size):
    # modify video paths and pretrain setting.
    cfg.CONFIG.DATA.VAL_DATA_PATH = UCF_DATA_ROOT  # 切换到对应的dataset   UCF_DATA_ROOT  Kinetic_DATA_ROOT
    cfg.CONFIG.DATA.VAL_ANNO_PATH = './ucf_all_info.csv'
    # cfg.CONFIG.DATA.VAL_ANNO_PATH = 'kinetics400_attack_samples.csv'  # 切换到对应的CSV文件   UCF_DATA_ROOT  Kinetic_DATA_ROOT
    cfg.CONFIG.MODEL.PRETRAINED = True
    cfg.CONFIG.VAL.BATCH_SIZE = batch_size
    return cfg

def get_cfg_custom(cfg_path, batch_size=16):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_path)
    cfg = change_cfg(cfg, batch_size)
    return cfg

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