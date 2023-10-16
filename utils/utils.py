import logging
import os
import random
from collections import defaultdict
from functools import partial
import piq

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


logger = logging.getLogger("logger")

def dict_html(dict_obj, current_time):
    out = ''
    for key, value in dict_obj.items():

        #filter out not needed parts:
        if key in ['poisoning_test', 'test_batch_size', 'discount_size', 'folder_path', 'log_interval',
                   'coefficient_transfer', 'grad_threshold' ]:
            continue

        out += f'<tr><td>{key}</td><td>{value}</td></tr>'
    output = f'<h4>Params for model: {current_time}:</h4><table>{out}</table>'
    return output


def init_seed(seed, reproducibility):
    r""" init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    

class ImageQuality:
    def __init__(self, metric_list, data_range=1.):
        self.metric2fuc = dict()
        self.metric2val = defaultdict(int)
        for metric in metric_list:
            if metric == "lpips":
                self.metric2fuc[metric] = eval("piq.LPIPS()")
            else:
                self.metric2fuc[metric] = partial(eval(f"piq.{metric}"), data_range=data_range)
        self.count = 0
        
    def compute(self, data_ori, data_tar):
        metric2val = defaultdict(int)
        with torch.no_grad():
            for metric, fuc in self.metric2fuc.items():
                metric2val[metric] = fuc(data_ori, data_tar).item()
                self.metric2val[metric] += metric2val[metric] * len(data_ori)
        self.count += len(data_ori)
        return metric2val
    
    def get_average(self):
        metric2val = defaultdict(int)
        for metric, val in self.metric2val.items():
            metric2val[metric] = val / self.count
        return metric2val
    
    def reset(self):
        self.metric2val = defaultdict(int)
        self.count = 0


def combine_dataset(dataloader_list):
    x_list, y_list = [], []
    for dataloader in dataloader_list:
        for batch_x, y in dataloader:
            x_list.append(batch_x)
            y_list.append(y)
    x_tensor = torch.cat(x_list)
    y_tensor = torch.cat(y_list)
    batch_size = dataloader_list[0].batch_size
    return DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=batch_size)


def attention2mask(attention, threshold=None, top_k=None):
    assert threshold or top_k
    if threshold:
        mask = (attention >= threshold).astype(int)
    else:
        # Step 1: Flatten the array
        flat_att = attention.flatten()

        # Step 2: Get the indices of the k largest values
        indices_of_top_k = np.argpartition(flat_att, -top_k)[-top_k:]

        # Step 3: Create a new array of zeros
        mask = np.zeros_like(flat_att)

        # Step 4: Set the top k positions to 1
        mask[indices_of_top_k] = 1

        # Reshape the array to its original shape
        mask = mask.reshape(attention.shape)
    return mask


def apply_grad_mask(model, mask_grad_list):
    mask_grad_list_copy = iter(mask_grad_list)
    for name, parms in model.named_parameters():
        if parms.requires_grad:
            parms.grad = parms.grad * next(mask_grad_list_copy)


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
