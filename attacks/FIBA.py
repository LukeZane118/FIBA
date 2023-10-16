from __future__ import annotations

import logging
from collections import deque
from math import exp
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import piq
import seaborn as sns
import torch
from torch.utils.data import DataLoader, TensorDataset
from utils.grad_cam import GradCam

from .base import Attack


logger = logging.getLogger("logger")
__all__ = ["FIBA", "train_trigger"]


class FIBA(Attack):
    """FIBA method to add trigger.
    """

    def __init__(
        self,
        init_magnitude: float = 0.01,
        pattern_shape: tuple = (3, 32, 32),
        mask: torch.Tensor | None = None,
    ):
        """
        Args:
            init_magnitude (float, optional): _description_. Defaults to 0.01.
            shape (tuple, optional): _description_. Defaults to (3, 32, 32).
            mask (torch.Tensor | None, optional): _description_. Defaults to None.
        """
        super().__init__()
        self.pattern_shape = pattern_shape
        self.pattern = torch.nn.Parameter(
            torch.rand(pattern_shape) * init_magnitude)
        self.mask = torch.nn.Parameter(
            mask, requires_grad=False) if mask is not None else None
        self.pattern_test = None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Add the trigger to images.

        Args:
            images (torch.Tensor): images to be added trigger.

        Returns:
            torch.Tensor: images with trigger.
        """
        if self.train:
            pattern = self.get_pattern()
        else:
            if self.pattern_test is None:
                self.pattern_test = self.get_pattern()
            pattern = self.pattern_test
        return torch.clamp(images + pattern, min=0.0, max=1.0)

    def get_pattern(self) -> torch.Tensor:
        """Get pattern of trigger.

        Returns:
            torch.Tensor: pattern of trigger.
        """
        return self.pattern if self.mask is None else self.pattern * self.mask

    def eval(self):
        self.pattern_test = None
        return super().eval()

    @torch.no_grad()
    def set_by_combination_(self, triggers: List[FIBA] | None, reduction: str = "add", device: torch.device | None = None) -> None:
        """Set up by merging triggers.

        Args:
            triggers (List[FIBA] | None): triggers to set up.
            reduction (str, optional): method of reduction. Defaults to "add".
            device (torch.device, optional): device of tensor. Defaults to None.
        """
        if triggers is None or len(triggers) == 0:
            return False

        self.pattern_shape = triggers[0].pattern_shape
        self.pattern = torch.nn.Parameter(sum(
            trigger.pattern * (1 if trigger.mask is None else trigger.mask) for trigger in triggers))

        if reduction == "mean" and triggers is not None and len(triggers) > 0:
            self.pattern /= len(triggers)

        if device:
            self.to(device)

        return True


def train_trigger(
    trigger: FIBA,
    model,
    y_target,
    device,
    image_shape=(3, 32, 32),
    lr=1e-3,
    reg_max=1.,
    reg_wmse=0,
    target_QoE=None,
    n_iter=200,
    threshold=1e-3,
    target_layers=None,
    use_mean_att=False,
    att_fig_save_path=None,
    dataloader=None,
    verbose=0
):
    # set model to eval
    model.eval()
    model.to(device)

    # set trigger to train
    trigger.train()
    trigger.to(device)

    # check parameter
    assert dataloader is not None
    # construct space attention map
    x_list, y_list = [], []
    for batch_x, y in dataloader:
        x_list.append(batch_x)
        y_list.append(y)
    x_tensor = torch.cat(x_list)
    y_tensor = torch.cat(y_list)
    batch_size = dataloader.batch_size
    dataloader_tmp = DataLoader(TensorDataset(
        x_tensor, y_tensor), batch_size=batch_size)
    space_att = get_weight_for_wmse(
        model, 
        dataloader_tmp, 
        device, 
        target_layers, 
        shape=image_shape[1:], 
        use_mean_att=use_mean_att, 
        att_fig_save_path=att_fig_save_path
        )
    img2att = torch.FloatTensor(space_att)
    del dataloader_tmp
    y_poi_tensor = torch.full_like(y_tensor, y_target)
    dataloader = DataLoader(TensorDataset(
        x_tensor, y_poi_tensor, img2att), batch_size=batch_size, shuffle=True)

    # train trigger
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(trigger.parameters(), lr=lr)
    n_iter_cap = n_iter << 1
    last_delta = None

    ssim_cal = piq.ssim
    psnr_cal = piq.psnr
    wmse_cal = WMSE()

    ssim_queue = deque(maxlen=10)

    # use PID control to get target ssim
    if target_QoE:
        PID = PIDControl(reg_wmse, W_MAX=reg_max)

    epoch = 0
    while True:
        total_loss = 0.
        total_loss_tri = 0.
        total_loss_ssim = 0.
        total_loss_psnr = 0.
        total_loss_wmse = 0.
        total_loss_norm = 0.
        n_batch = 0
        for batch_x, batch_y, batch_att in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_att = batch_att.to(device)
            optimizer.zero_grad(set_to_none=True)
            model.zero_grad(set_to_none=True)
            batch_x_pos = trigger(batch_x)
            output = model(batch_x_pos)

            loss_tri = criterion(output, batch_y)
            loss_ssim = -ssim_cal(batch_x, batch_x_pos)
            loss_psnr = -psnr_cal(batch_x, batch_x_pos)
            loss_wmse = wmse_cal(batch_x, batch_x_pos, batch_att)
            loss_norm = torch.abs(trigger.get_pattern()).mean()
            loss = loss_tri + reg_wmse * loss_wmse
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_loss_tri += loss_tri.item()
            total_loss_ssim += loss_ssim.item()
            total_loss_psnr += loss_psnr.item()
            total_loss_wmse += loss_wmse.item()
            total_loss_norm += loss_norm.item()
            n_batch += 1

        mean_loss_ssim = total_loss_ssim / n_batch

        ssim_queue.append(mean_loss_ssim)

        ssim_std = np.std(ssim_queue)

        is_convergent = (len(ssim_queue) == ssim_queue.maxlen) and (
            ssim_std < 1e-3)

        if verbose > 0 and (epoch + 1) % verbose == 0:
            logger.info(f"Epoch {epoch}: total loss: {total_loss / n_batch}, trigger loss: {total_loss_tri / n_batch}, "
                        f"ssim loss: {total_loss_ssim / n_batch}, psnr loss: {total_loss_psnr / n_batch}, "
                        f"wmse loss: {total_loss_wmse / n_batch}, norm loss: {total_loss_norm / n_batch}")

        if target_QoE and is_convergent:
            reg_wmse = PID.pid(-target_QoE, mean_loss_ssim)
            ssim_queue.clear()
            if target_QoE:
                logger.info(f"Weight of wmse change to {reg_wmse}")

        if epoch + 1 >= n_iter:
            if target_QoE is None:
                logger.info(f"Last average ssim: {mean_loss_ssim}")
                break
            cur_delta = abs(mean_loss_ssim + target_QoE)
            # convergent and reaches target
            if is_convergent and cur_delta < threshold:
                logger.info(f"Last average ssim: {mean_loss_ssim}")
                break
            elif epoch + 1 >= n_iter_cap:
                # not convergent but is convergening
                if last_delta is None or cur_delta <= last_delta:
                    if n_iter_cap >= 10 * n_iter:  # maximum cap
                        logger.info(f"Last average ssim: {mean_loss_ssim}")
                        break
                    n_iter_cap += n_iter
                    last_delta = cur_delta
                # not convergetent and not convergening (it means we should adjust hyperparameters)
                else:
                    logger.info(f"Last average ssim: {mean_loss_ssim}")
                    break

        epoch += 1

    trigger.eval()

    trigger.zero_grad()
    model.zero_grad()

    # set model to train
    model.train()

    return trigger


def get_weight_for_wmse(model, dataloader, device, target_layers, shape=(32, 32), use_mean_att=False, att_fig_save_path=None):
    w, h = dataloader.dataset.tensors[0].shape[-2:]
    if target_layers is None or len(target_layers) == 0:
        return np.ones((len(dataloader.dataset), 1, w, h)) / (w * h)
    model.eval()
    gcam = GradCam(model, device, target_layers, shape[-2:])
    space_attention = np.empty(
        (len(dataloader.dataset), 1, w, h))  # b * c * w * h
    i = 0
    for batch_x, _ in dataloader:
        batch_x = batch_x.to(device)
        for j in range(len(batch_x)):
            attention, _ = gcam(batch_x[j:j + 1])
            attention = 1 - attention
            space_attention[i][0] = attention / max(np.sum(attention), 1.e-8)
            i += 1
        # pbar.update()
    model.zero_grad(set_to_none=True)
    logger.info("Got weight for wmse.")
    if use_mean_att:
        logger.info("Use mean weight for wmse.")
        space_attention = np.mean(space_attention, axis=0, keepdims=True)
        space_attention = space_attention / \
            np.max(np.sum(space_attention), 1.e-8)
        if att_fig_save_path:
            sns.heatmap(
                space_attention.numpy(),
                vmin=0.0,
                vmax=1.0,
                cmap="jet",
                cbar=True,
                xticklabels=False,
                yticklabels=False,
            )
            plt.savefig(att_fig_save_path)
            plt.close("all")
            logger.info("Save mean weight to picture.")
    return space_attention


class WMSE:
    def __init__(self):
        self.name = "WMSE"

    @staticmethod
    def __call__(image1, image2, w=None):
        if w is None:
            w = 1 / (image1.shape[-2] * image1.shape[-1])
        return torch.mean(torch.sum((image1 - image2) ** 2 * w, dim=(-2, -1)))


class PIDControl():
    """docstring for ClassName"""

    def __init__(self, W_k0=0., W_MAX=1.):
        """define them out of loop"""
        # self.exp_KL = exp_KL
        self.W_MAX = W_MAX
        self.W_k0 = W_k0 / W_MAX
        self.I_k1 = 0.0
        self.W_k1 = 0.0
        self.e_k1 = 0.0

    def _Kp_fun(self, Err, scale=1):
        return 1.0/(1.0 + float(scale)*exp(Err))

    def pid(self, exp_val, cur_val, Kp=0.01, Ki=-1, Kd=0.01):  # Kp: 0.001, Ki: -0.2 for wmse
        """
        position PID algorithm
        Input: exp_val, Ki (speed of adjusting) 
        return: weight for cur_val
        """
        error_k = exp_val - cur_val
        # comput U as the control factor
        Pk = Kp * self._Kp_fun(error_k)
        Ik = self.I_k1 + Ki * error_k
        # Dk = (error_k - self.e_k1) * Kd

        # window up for integrator
        if self.W_k1 < 0 and self.W_k1 >= 1:
            Ik = self.I_k1

        Wk = Pk + Ik + self.W_k0
        self.W_k1 = Wk
        self.I_k1 = Ik
        self.e_k1 = error_k

        # min and max value
        if Wk > 1:
            Wk = 1.0
        if Wk < 0:
            Wk = 0.0

        return Wk * self.W_MAX
