import numpy as np
import torch

from .ConvPkg import conv2d
from typing import Union

def sobel(numpy_array=False, in_channels=1, out_channels=1):
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]])
    
    kernel_x = np.expand_dims(kernel_x, axis=(0, 1))
    kernel_y = np.expand_dims(kernel_y, axis=(0, 1))
    kernel_x = np.repeat(kernel_x, in_channels, axis=1)
    kernel_x = np.repeat(kernel_x, out_channels, axis=0)
    kernel_y = np.repeat(kernel_y, in_channels, axis=1)
    kernel_y = np.repeat(kernel_y, out_channels, axis=0)

    if numpy_array:
        return kernel_x, kernel_y    
    return torch.from_numpy(kernel_x).float(), torch.from_numpy(kernel_y).float()
    
def grad(image:Union[torch.Tensor, np.ndarray]):
    if isinstance(image, np.ndarray):
        if image.ndim == 3:
            image = torch.from_numpy(image).unsqueeze(0)

    sobel_kernel_x, sobel_kernel_y = sobel(True, in_channels=1, out_channels=image.shape[1])
    grad_map_x = conv2d(image, sobel_kernel_x, groups=image.shape[1])
    grad_map_y = conv2d(image, sobel_kernel_y, groups=image.shape[1])
    return grad_map_x, grad_map_y

def Laplacian(numpy_array=False, in_channels=1, out_channels=1):
    kernel = np.array([[0, 1, 0],
                         [1, -4, 1],
                         [0, 1, 0]])
    kernel = np.expand_dims(kernel, axis=(0, 1))
    kernel = np.repeat(kernel, in_channels, axis=1)
    kernel = np.repeat(kernel, out_channels, axis=0)
        
    if not numpy_array:
        return torch.from_numpy(kernel).float()
    return kernel
