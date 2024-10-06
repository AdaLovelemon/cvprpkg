from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import numpy as np

from typing import Union, Optional

def read_img(img_path:str, float_type:bool=False, color:bool=True, numpy_array:bool=False):
    if color:
        transform = transforms.Compose([transforms.ToTensor()])  # 将图像转换为Tensor
                        
    else:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # 将图像转换为单通道灰度图像
            transforms.ToTensor()  # 将图像转换为Tensor
            ])
        
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0)
    if float_type:
        return img.to(torch.float32)[0].numpy() if numpy_array else img.to(torch.float32) 
    img = (img * 255).byte()
    return img[0].numpy() if numpy_array else img


def imshow(img:Union[torch.Tensor, np.ndarray], color:Optional[bool]=None, save_path:Optional[str]=None):
    numpy_array = isinstance(img, np.ndarray)

    if not numpy_array:
        if color is None:
            color = (img.size(1) > 1)
        if color:
            img = img[0].permute(1, 2, 0).numpy()
            plt.imshow(img)
        else:
            img = img[0, 0].numpy()
            plt.imshow(img, cmap='gray')
    else:
        if color is None:
            color = (img.shape[0] > 1)
        if color:
            img = img.transpose(1, 2, 0)
            plt.imshow(img)
        else:
            img = img[0]
            plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if save_path is not None:
        format = save_path.split('.')[-1]
        plt.savefig(save_path, format=format, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.show()