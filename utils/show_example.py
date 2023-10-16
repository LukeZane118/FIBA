import functools
import operator
import os
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import torch
import cv2
from utils.grad_cam import GradCam


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.5) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}")

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def show_images(images: torch.Tensor, subtitle = None, suptitle = None, save_path = None):
    # if type(images[0]) != list:
    #     images = [images]
    n_rows = len(images)
    n_columns = len(images[0])
    # n_rows * n_columns = len(subtitle)
    fig = plt.figure(figsize=(2 * n_columns, 2 * n_rows), dpi=150)
    
    if save_path is not None:
        singe_image_path = os.path.join(save_path, "single_example")
        os.makedirs(singe_image_path)
    else:
        singe_image_path = None

    for i, images_row in enumerate(images):
        for j, image in enumerate(images_row):
            ax = fig.add_subplot(n_rows, n_columns, i * n_columns + j + 1)
            # plt.axis('off')
            plt.box(False)
            plt.xticks([])
            plt.yticks([])
            if type(image) == torch.Tensor:
                image = image.permute((1, 2, 0)).cpu().detach().numpy()
            image = image.squeeze()
            if singe_image_path is not None:
                mp.image.imsave(os.path.join(singe_image_path, f"example_r{i}_c{j}.png"), image)
            if len(image.shape) < 3:
                plt.imshow(image, cmap='gray')
            else:
                plt.imshow(image)

    axes = fig.axes
    
    if subtitle is not None:
    
        subtitle = functools.reduce(operator.iconcat, subtitle, [])
        
        for ax, title in zip(axes[:len(subtitle)], subtitle):
            ax.set_title(title)
        
    # for i, ax in enumerate(axes[::columns]):
    #     ax.set_ylabel(f'row{i}')
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    plt.show()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, "examples.png"))
        print("Save Examples.")
    plt.close()


def generate_rows(model, trigger, target_layers, images, labels, target_label, device):
    model.eval()
    trigger.eval()
    rows, titles = [], []
    # img_sample = np.random.choice(len(y_test), size=n_imgs, replace=False)
    gcam = GradCam(model, device, target_layers, size=images.shape[2:])
    for img, label in zip(images, labels):
        trigger.zero_grad()
        model.zero_grad()
        x_cle = img[None].to(device)
        x_pos = trigger(x_cle)
        mask_cle, yp_cle = gcam(x_cle)
        mask_pos, yp_pos = gcam(x_pos)
        img_np = img.cpu().numpy().transpose((1, 2, 0))
        visualization_cle = show_cam_on_image(img_np, mask_cle, use_rgb=True)
        visualization_pos = show_cam_on_image(img_np, mask_pos, use_rgb=True)
        rows.append([x_cle[0], x_pos[0], visualization_cle, visualization_pos])
        titles.append([f"(label {label})", f"(target label {target_label})", f"(predict {yp_cle})", f"(predict {yp_pos})"])
    trigger.zero_grad()
    model.zero_grad()
    return rows, titles

def generate_rows_general(model, poi_fuc, target_layers, images, labels, target_label, device):
    model.eval()
    rows, titles = [], []
    # img_sample = np.random.choice(len(y_test), size=n_imgs, replace=False)
    gcam = GradCam(model, device, target_layers, size=images.shape[2:])
    for img, label in zip(images, labels):
        model.zero_grad()
        # x_cle = img[None].to(device)
        x_pos, _, _ = poi_fuc((img[None], label[None]), evaluation=True)
        mask_cle, yp_cle = gcam(img[None].to(device))
        mask_pos, yp_pos = gcam(x_pos[None])
        img_np = img.cpu().numpy().transpose((1, 2, 0))
        visualization_cle = show_cam_on_image(img_np, mask_cle, use_rgb=True)
        visualization_pos = show_cam_on_image(img_np, mask_pos, use_rgb=True)
        rows.append([img, x_pos, visualization_cle, visualization_pos])
        titles.append([f"(label {label})", f"(target label {target_label})", f"(predict {yp_cle})", f"(predict {yp_pos})"])
    model.zero_grad()
    return rows, titles