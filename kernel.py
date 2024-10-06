import numpy as np 
import torch
from matplotlib import pyplot as plt

from typing import Union, List, Tuple

from .ConvPkg import _padding2d

def GaussianKernel2d(kernel_size:Union[List[int], Tuple[int, int], int], sigma:float, in_channels:int=1, out_channels:int=1, numpy_array:bool=False):
    # [out_channels, in_channels, kernel_height, kernel_width]
    if not isinstance(kernel_size, (list, tuple)):
        kernel_size = (kernel_size, kernel_size)
    kernel_radius = (kernel_size[0] // 2, kernel_size[1] // 2)
    x, y = np.meshgrid(np.arange(-kernel_radius[0], kernel_radius[0] + 1), np.arange(-kernel_radius[1], kernel_radius[1] + 1))
    
    gaussian_kernel2d = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    gaussian_kernel2d = np.expand_dims(gaussian_kernel2d, axis=(0, 1))
    gaussian_kernel2d = np.repeat(gaussian_kernel2d, in_channels, axis=1)
    gaussian_kernel2d = np.repeat(gaussian_kernel2d, out_channels, axis=0)
    gaussian_kernel2d = gaussian_kernel2d / np.sum(gaussian_kernel2d, axis=(1, 2, 3), keepdims=True)
    
    return torch.from_numpy(gaussian_kernel2d).float() if not numpy_array else gaussian_kernel2d

def GaussianKernel1d(kernel_size:Union[list, tuple, int], sigma:float, in_channels:int=1, out_channels:int=1, numpy_array:bool=False):
    # [out_channels, in_channels, kernel_len]
    if not(isinstance(kernel_size, list) or isinstance(kernel_size, tuple)):
        kernel_size = (kernel_size,)
    kernel_radius = kernel_size[0] // 2
    x = np.arange(-kernel_radius, kernel_radius + 1)
    gaussian_kernel1d = np.exp(-(x**2) / (2 * sigma**2))
    
    gaussian_kernel1d = np.expand_dims(gaussian_kernel1d, axis=(0, 1))
    gaussian_kernel1d = np.repeat(gaussian_kernel1d, in_channels, axis=1)
    gaussian_kernel1d = np.repeat(gaussian_kernel1d, out_channels, axis=0)
    gaussian_kernel1d /= np.sum(gaussian_kernel1d, axis=(1, 2, 3), keepdims=True)

    return gaussian_kernel1d if numpy_array else torch.from_numpy(gaussian_kernel1d).float()
    
def PixelGaussianKernel2d(image:np.ndarray, kernel_size:Union[int, list, tuple], sigma:float, centroid_x:int, centroid_y:int):
    """
    Image Shape should be [num_channels, height, width]
    """
    if not isinstance(kernel_size, (list, tuple)):
        kernel_size = (kernel_size, kernel_size)
    kernel_radius = (kernel_size[0] // 2, kernel_size[1] // 2)
    x_low, x_high, y_low, y_high = -kernel_radius[0] + centroid_x, kernel_radius[0] + 1 + centroid_x, -kernel_radius[1] + centroid_y, kernel_radius[1] + 1 + centroid_y
    image_patch = image[:, x_low:x_high, y_low:y_high]
    kernel = np.exp(-np.abs(image_patch - image[:, centroid_x, centroid_y].reshape(-1, 1, 1)) / (2 * sigma**2))
    kernel /= np.sum(kernel, axis=(1, 2), keepdims=True)

    return kernel
    
def BilateralFilter2d(image:Union[torch.Tensor, np.ndarray], kernel_size:Union[list, tuple, int], sigma_s:float, sigma_r:float, padding='valid', padding_mode='reflect'):
    image = _padding2d(image, padding, (image.shape[0], image.shape[0], kernel_size, kernel_size), padding_mode)
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    image = image[0]
    DistGaussianKernel = GaussianKernel2d(kernel_size, sigma_s, numpy_array=True)[0]
    feature_map = np.zeros((image.shape[0], image.shape[1] - kernel_size + 1, image.shape[2] - kernel_size + 1))
    for x in range(image.shape[1] - kernel_size + 1):
        for y in range(image.shape[2] - kernel_size + 1):
            kernel_radius = kernel_size // 2
            centroid_x, centroid_y = x + kernel_radius, y + kernel_radius
            PixelGaussianKernel = PixelGaussianKernel2d(image, kernel_size, sigma_r, centroid_x, centroid_y)
            kernel = DistGaussianKernel * PixelGaussianKernel
            kernel /= np.sum(kernel, axis=(1, 2), keepdims=True)
            feature_map[:, x, y] = (kernel * image[:, x:x+kernel_size, y:y+kernel_size]).sum()

    return feature_map

def KernelDisplay(kernel: Union[torch.Tensor, np.ndarray], cmap=None):
    dim1 = False
    if kernel.ndim == 3:
        kernel = kernel.reshape(kernel.shape[0], kernel.shape[1], 1, kernel.shape[2])
        dim1 = True
    # 确保 kernel 是 NumPy 数组
    if isinstance(kernel, torch.Tensor):
        kernel = kernel.detach().cpu().numpy()

    # 获取卷积核的形状 (out_channels, in_channels, height, width)
    out_channels, in_channels, height, width = kernel.shape

    # 创建子图网格，行数为输出通道数，列数为输入通道数
    fig, axes = plt.subplots(out_channels, in_channels, figsize=(in_channels * 2, out_channels * 2))

    # 遍历每个输出通道和输入通道，展示对应的卷积核
    for i in range(out_channels):
        for j in range(in_channels):
            # 选择特定的卷积核 (i, j) 对应的子矩阵
            current_kernel = kernel[i, j]

            # 如果只有一个通道，axes 不是二维数组，需要特殊处理
            if out_channels == 1 and in_channels == 1:
                axes.matshow(current_kernel, cmap=cmap)
                if dim1: axes.set_yticks([])
            elif out_channels == 1 or in_channels == 1:
                axes[max(i,j)].matshow(current_kernel, cmap=cmap)
                if dim1: axes[max(i, j)].set_yticks([])
            else:
                axes[i, j].matshow(current_kernel, cmap=cmap)
                if dim1:
                    axes[i, j].set_yticks([])
    # 自动调整布局并显示
    plt.tight_layout()
    plt.show()


def DerivativeGaussianKernel2d(kernel_size:Union[List[int], Tuple[int, int], int], sigma:float, in_channels:int=1, out_channels:int=1, numpy_array:bool=False):
    if not isinstance(kernel_size, (list, tuple)):
        kernel_size = (kernel_size, kernel_size)
    kernel_radius = (kernel_size[0] // 2, kernel_size[1] // 2)
    x, y = np.meshgrid(np.arange(-kernel_radius[0], kernel_radius[0] + 1), np.arange(-kernel_radius[1], kernel_radius[1] + 1))
    
    # Numerical errors would happen if sigma >> kernel_size, which means the kernel is almost the sobel, summing up to 0
    D_gaussian_kernel2d_x = np.exp(-(x**2 + y**2) / (2 * sigma**2)) * x
    D_gaussian_kernel2d_y = np.exp(-(x**2 + y**2) / (2 * sigma**2)) * y
    if np.sum(D_gaussian_kernel2d_x) != 0:
        D_gaussian_kernel2d_x /= np.sum(D_gaussian_kernel2d_x)
    if np.sum(D_gaussian_kernel2d_y) != 0:
        D_gaussian_kernel2d_y /= np.sum(D_gaussian_kernel2d_y)

    D_gaussian_kernel2d_x = np.expand_dims(D_gaussian_kernel2d_x, axis=(0, 1))
    D_gaussian_kernel2d_x = np.repeat(D_gaussian_kernel2d_x, in_channels, axis=1)
    D_gaussian_kernel2d_x = np.repeat(D_gaussian_kernel2d_x, out_channels, axis=0)
    D_gaussian_kernel2d_y = np.expand_dims(D_gaussian_kernel2d_y, axis=(0, 1))
    D_gaussian_kernel2d_y = np.repeat(D_gaussian_kernel2d_y, in_channels, axis=1)
    D_gaussian_kernel2d_y = np.repeat(D_gaussian_kernel2d_y, out_channels, axis=0)

    if numpy_array:
        return -D_gaussian_kernel2d_x, -D_gaussian_kernel2d_y
    return -torch.from_numpy(D_gaussian_kernel2d_x).float(), -torch.from_numpy(D_gaussian_kernel2d_y).float()

    