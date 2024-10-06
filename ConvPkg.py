import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from typing import Optional, Union
import math

def padding1d(X: torch.Tensor, padding: Union[tuple, list], padding_mode: str='zero', constant_val=0):
    if not (padding_mode in ('zero', 'constant', 'wrap', 'copy', 'reflect')):
        raise ValueError("Invalid padding_mode!")

    if not(padding_mode == 'constant') and constant_val != 0:
        raise ValueError("Only Constant Mode needs to input `constant_val`")  

    # Unpack the padding values
    padding_left, padding_right = padding

    batch_size, in_channels, length = X.shape
    padded_length = length + padding_left + padding_right
    
    if padding_mode == 'zero' or padding_mode == 'constant':
        if padding_mode == 'zero':
            constant_val = 0
        X_pad = torch.full((batch_size, in_channels, padded_length), constant_val, device=X.device, dtype=X.dtype)
        X_pad[:, :, padding_left:padding_left+length] = X
    
    elif padding_mode == 'wrap':
        X_pad = torch.zeros((batch_size, in_channels, padded_length), device=X.device, dtype=X.dtype)
        X_pad[:, :, padding_left:padding_left+length] = X
        X_pad[:, :, :padding_left] = X[:, :, -padding_left:]
        X_pad[:, :, -padding_right:] = X[:, :, :padding_right]

    elif padding_mode == 'copy':
        X_pad = torch.zeros((batch_size, in_channels, padded_length), device=X.device, dtype=X.dtype)
        X_pad[:, :, padding_left:padding_left+length] = X
        X_pad[:, :, :padding_left] = X[:, :, 0:1].expand(-1, -1, padding_left)
        X_pad[:, :, -padding_right:] = X[:, :, -1:].expand(-1, -1, padding_right)

    elif padding_mode == 'reflect':
        X_pad = torch.zeros((batch_size, in_channels, padded_length), device=X.device, dtype=X.dtype)
        X_pad[:, :, padding_left:padding_left+length] = X
        X_pad[:, :, :padding_left] = X[:, :, :padding_left].flip(2)
        X_pad[:, :, -padding_right:] = X[:, :, -padding_right:].flip(2)

    return X_pad

def padding2d(X: torch.Tensor, padding: Union[tuple, list], padding_mode: str='zero', constant_val=0):
    if not (padding_mode in ('zero', 'constant', 'wrap', 'copy', 'reflect')):
        raise ValueError("Invalid padding_mode!")

    if not(padding_mode == 'constant') and constant_val != 0:
        raise ValueError("Only Constant Mode needs to input `constant_val`")  

    # Unpack the padding values
    padding_top, padding_bottom, padding_left, padding_right = padding

    batch_size, in_channels, height, width = X.shape
    padded_height = height + padding_top + padding_bottom
    padded_width = width + padding_left + padding_right
    
    if padding_mode == 'zero' or padding_mode == 'constant':
        if padding_mode == 'zero':
            constant_val = 0
        X_pad = torch.full((batch_size, in_channels, padded_height, padded_width), constant_val, device=X.device, dtype=X.dtype)
        X_pad[:, :, padding_top:padding_top+height, padding_left:padding_left+width] = X
    
    elif padding_mode == 'wrap':
        X_pad = torch.zeros((batch_size, in_channels, padded_height, padded_width), device=X.device, dtype=X.dtype)
        X_pad[:, :, padding_top:padding_top+height, padding_left:padding_left+width] = X
        X_pad[:, :, :padding_top, padding_left:padding_left+width] = X[:, :, -padding_top:, :]
        X_pad[:, :, -padding_bottom:, padding_left:padding_left+width] = X[:, :, :padding_bottom, :]
        X_pad[:, :, padding_top:padding_top+height, :padding_left] = X[:, :, :, -padding_left:]
        X_pad[:, :, padding_top:padding_top+height, -padding_right:] = X[:, :, :, :padding_right]
        X_pad[:, :, :padding_top, :padding_left] = X[:, :, -padding_top:, -padding_left:]
        X_pad[:, :, :padding_top, -padding_right:] = X[:, :, -padding_top:, :padding_right]
        X_pad[:, :, -padding_bottom:, :padding_left] = X[:, :, :padding_bottom, -padding_left:]
        X_pad[:, :, -padding_bottom:, -padding_right:] = X[:, :, :padding_bottom, :padding_right]

    elif padding_mode == 'copy':
        X_pad = torch.zeros((batch_size, in_channels, padded_height, padded_width), device=X.device, dtype=X.dtype)
        X_pad[:, :, padding_top:padding_top+height, padding_left:padding_left+width] = X
        X_pad[:, :, :padding_top, padding_left:padding_left+width] = X[:, :, 0:1, :].expand(-1, -1, padding_top, -1)
        X_pad[:, :, -padding_bottom:, padding_left:padding_left+width] = X[:, :, -1:, :].expand(-1, -1, padding_bottom, -1)
        X_pad[:, :, padding_top:padding_top+height, :padding_left] = X[:, :, :, 0:1].expand(-1, -1, -1, padding_left)
        X_pad[:, :, padding_top:padding_top+height, -padding_right:] = X[:, :, :, -1:].expand(-1, -1, -1, padding_right)
        X_pad[:, :, :padding_top, :padding_left] = X[:, :, 0:1, 0:1].expand(-1, -1, padding_top, padding_left)
        X_pad[:, :, :padding_top, -padding_right:] = X[:, :, 0:1, -1:].expand(-1, -1, padding_top, padding_right)
        X_pad[:, :, -padding_bottom:, :padding_left] = X[:, :, -1:, 0:1].expand(-1, -1, padding_bottom, padding_left)
        X_pad[:, :, -padding_bottom:, -padding_right:] = X[:, :, -1:, -1:].expand(-1, -1, padding_bottom, padding_right)

    elif padding_mode == 'reflect':
        X_pad = torch.zeros((batch_size, in_channels, padded_height, padded_width), device=X.device, dtype=X.dtype)
        X_pad[:, :, padding_top:padding_top+height, padding_left:padding_left+width] = X
        X_pad[:, :, :padding_top, padding_left:padding_left+width] = X[:, :, :padding_top, :].flip(2)
        X_pad[:, :, -padding_bottom:, padding_left:padding_left+width] = X[:, :, -padding_bottom:, :].flip(2)
        X_pad[:, :, padding_top:padding_top+height, :padding_left] = X[:, :, :, :padding_left].flip(3)
        X_pad[:, :, padding_top:padding_top+height, -padding_right:] = X[:, :, :, -padding_right:].flip(3)
        X_pad[:, :, :padding_top, :padding_left] = X[:, :, :padding_top, :padding_left].flip(2, 3)
        X_pad[:, :, :padding_top, -padding_right:] = X[:, :, :padding_top, -padding_right:].flip(2, 3)
        X_pad[:, :, -padding_bottom:, :padding_left] = X[:, :, -padding_bottom:, :padding_left].flip(2, 3)
        X_pad[:, :, -padding_bottom:, -padding_right:] = X[:, :, -padding_bottom:, -padding_right:].flip(2, 3)

    return X_pad

def calculate_same_padding1d(length, kernel_size, stride, dilation):
    kernel_length = kernel_size[2]
    stride_length = stride[0]
    dilation_length = dilation[0]
    
    # 计算有效卷积核大小
    effective_kernel_length = (kernel_length - 1) * dilation_length + 1
    
    # 计算总的填充大小
    padding_length = max(0, (length - 1) * stride_length + effective_kernel_length - length)
    
    # 计算每边的填充大小
    padding_left = padding_length // 2
    padding_right = padding_length - padding_left
    
    return (padding_left, padding_right)

def calculate_same_padding2d(height, width, kernel_size, stride, dilation):
    kernel_height, kernel_width = kernel_size[2], kernel_size[3]
    stride_height, stride_width = stride
    dilation_height, dilation_width = dilation
    
    # 计算有效卷积核大小
    effective_kernel_height = (kernel_height - 1) * dilation_height + 1
    effective_kernel_width = (kernel_width - 1) * dilation_width + 1
    
    # 计算总的填充大小
    padding_height = max(0, (height - 1) * stride_height + effective_kernel_height - height)
    padding_width = max(0, (width - 1) * stride_width + effective_kernel_width - width)
    
    # 计算每边的填充大小
    padding_top = padding_height // 2
    padding_bottom = padding_height - padding_top
    padding_left = padding_width // 2
    padding_right = padding_width - padding_left
    
    return (padding_top, padding_bottom, padding_left, padding_right)

def _padding1d(X: torch.Tensor, padding: Union[str, int, list, tuple], kernel_size: Union[list, tuple], padding_mode: str='zero', constant_val=0, dilation: Union[list, tuple]=(1,), stride: Union[list, tuple]=(1,)):
    batch_size, in_channels, length = X.size()
    kernel_length = kernel_size[2]
    
    if padding == 'same':
        padding = calculate_same_padding1d(length, kernel_size, stride, dilation)
        # Apply the padding
        X_padding = padding1d(X, padding, padding_mode=padding_mode, constant_val=constant_val)
    
    elif padding == 'valid':
        X_padding = X

    elif padding == 'full':
        padding = (kernel_length - 1, kernel_length - 1)
        X_padding = padding1d(X, padding, padding_mode=padding_mode, constant_val=constant_val)

    else:
        if not (isinstance(padding, list) or isinstance(padding, tuple)):
            padding = (padding, padding)
        else:
            padding = (padding[0], padding[1])
        X_padding = padding1d(X, padding, padding_mode=padding_mode, constant_val=constant_val)

    return X_padding

def _padding2d(X:torch.Tensor, padding:Union[int, list, tuple], kernel_size:Union[list, tuple], padding_mode:str='zero', constant_val=0, dilation:Union[list, tuple]=(1,1), stride:Union[list, tuple]=(1,1)):
    batch_size, in_channels, height, width = X.size()
    kernel_height, kernel_width = kernel_size[2], kernel_size[3]
    
    if padding == 'same':
        padding = calculate_same_padding2d(height, width, kernel_size, stride, dilation)
        # Apply the padding
        X_padding = padding2d(X, padding, padding_mode=padding_mode, constant_val=constant_val)
    
    elif padding == 'valid':
        X_padding = X

    elif padding == 'full':
        padding = (kernel_height - 1, kernel_height - 1, kernel_width - 1, kernel_width - 1)
        X_padding = padding2d(X, padding, padding_mode=padding_mode, constant_val=constant_val)

    else:
        if not (isinstance(padding, list) or isinstance(padding, tuple)):
            padding = (padding, padding, padding, padding)
        else:
            padding = (padding[0], padding[0], padding[1], padding[1])
        X_padding = padding2d(X, padding, padding_mode=padding_mode, constant_val=constant_val)

    return X_padding

def expand_kernel(kernel: torch.Tensor, dilation: Union[int,list,tuple], dtype=None, device=None):
    '''
    Expand the kernel with certain dilation.
    ========================================

    Parameters
    ----------
    - kernel: The original conv kernel [out_channels, in_channels, kernel_height, kernel_width]. 1D or 2D.
    - dilation: The dilation, list or tuple or int.

    Return
    ------
    - Expansion of the kernel.
    '''
    if dtype is None:
        dtype = kernel.dtype
    if device is None:
        device = kernel.device

    if len(kernel.size()) == 4:
        if not (isinstance(dilation, list) or isinstance(dilation, tuple)):
            dilation = (dilation, dilation)

        out_channels, in_channels, kernel_height, kernel_width = kernel.size()
        expanded_height = kernel_height + (kernel_height - 1) * (dilation[0] - 1)
        expanded_width = kernel_width + (kernel_width - 1) * (dilation[1] - 1)
        
        expanded_kernel = torch.zeros((out_channels, in_channels, expanded_height, expanded_width), dtype=dtype, device=device)
        for i in range(kernel_height):
            for j in range(kernel_width):
                    expanded_kernel[:, :, i * dilation[0], j * dilation[1]] = kernel[:, :, i, j]
    elif len(kernel.size()) == 3:
        out_channels, in_channels, kernel_len = kernel.size()
        expanded_len = kernel_len + (kernel_len - 1) * (dilation - 1)

        expanded_kernel = torch.zeros((out_channels, in_channels, expanded_len), dtype=dtype, device=device)
        for i in range(kernel_len):
            expanded_kernel[:, :, i * dilation] = kernel[:, :, i]
    else:
        raise ValueError('Kernel shape invalid!')

    return expanded_kernel

def conv1d(X: torch.Tensor, kernel: torch.Tensor, bias: Optional[torch.Tensor] = None, stride: Union[int, list, tuple] = 1, padding: Union[int, list, tuple] = 0, padding_mode: str = 'zero', dilation: Union[int, list, tuple] = 1, constant_val: float = 0, is_correlation: bool = True):
    '''
    DIY Conv1d
    ========================================

    Parameters
    ----------
    - X: The input of the convolution [batch_size, in_channels, seq_len]
    - kernel: The original conv kernel [out_channels, in_channels, kernel_len]
    - bias: The bias to be added upon the output.
    - stride: The stride, list or tuple or int.
    - padding: The padding, list or tuple or int. If stride is set as `'same'`, then the convolution will compute in same padding mode. Additionally, padding can also be set as `'valid'` or `'full'`.
    - dilation: The dilation, list or tuple or int.
    - constant_val: The value used to do constant padding.
    - is_correlation: If true, the function will execute correlation operation; otherwise it will execute convolution.

    Return
    ------
    - Output of the convolution.

    Attention
    ---------
    - All the computation will be based on float64.

    '''
    batch_size, in_channels, seq_len = X.size()
    assert in_channels == kernel.size(1), 'The input channels do not match!'

    if not (isinstance(stride, list) or isinstance(stride, tuple)):
        stride = (stride,)

    if not is_correlation:
        kernel = torch.flip(kernel, [2])

    # Expand the kernel
    if not (isinstance(dilation, list) or isinstance(dilation, tuple)):
        dilation = (dilation,)
    if dilation[0] != 1:
        kernel_new = expand_kernel(kernel, dilation)
    else:
        kernel_new = kernel
    kernel_len = kernel_new.size(2)

    X_padding = _padding1d(X, padding, kernel.size(), padding_mode, constant_val, dilation, stride)    

    X_padding = X_padding.unfold(2, kernel_len, stride[0])

    # Unfold the input and start to sum
    out = torch.einsum('bilk,oik->bol', X_padding, kernel_new)
    if bias is not None:
        out = out + bias.view(1, -1, 1)

    return out

def conv2d(X: torch.Tensor, kernel: torch.Tensor, bias: Optional[torch.Tensor] = None, stride: Union[int, list, tuple] = 1, padding: Union[int, list, tuple] = 0, padding_mode: str = 'zero', dilation: Union[int, list, tuple] = 1, constant_val: float = 0, is_correlation: bool = True, groups:int=1):
    '''
    DIY Conv2d
    ========================================

    Parameters
    ----------
    - X: The input of the convolution [batch_size, in_channels, height, width]
    - kernel: The original conv kernel [out_channels, in_channels, kernel_height, kernel_width]
    - bias: The bias to be added upon the output. [batch_size, out_channels]
    - stride: The stride, list or tuple or int.
    - padding: The padding, list or tuple or int. If stride is set as `'same'`, then the convolution will compute in same padding mode. Additionally, padding can also be set as `'valid'` or `'full'`.
    - dilation: The dilation, list or tuple or int.
    - constant_val: The value used to do constant padding.
    - is_correlation: If true, the function will execute correlation operation; otherwise it will execute convolution.
    - groups: Number of blocked connections from input channels to output channels.

    Return
    ------
    - Output of the convolution.

    Attention
    ---------
    - All the computation will be based on float64.

    '''
    batch_size, in_channels, height, width = X.size()
    out_channels, kernel_in_channels = kernel.size(0), kernel.size(1)
    assert in_channels == kernel_in_channels * groups, 'The input channels does not match!'
    if not (isinstance(stride, list) or isinstance(stride, tuple)):
            stride = (stride, stride)

    if not is_correlation:
        kernel = torch.flip(kernel, [2, 3])

    # Expand the kernel
    if not (isinstance(dilation, list) or isinstance(dilation, tuple)):
        dilation = (dilation, dilation)
    if dilation[0] != 1 or dilation[1] != 1: 
        kernel_new = expand_kernel(kernel, dilation)
    else:
        kernel_new = kernel
    kernel_height, kernel_width = kernel_new.size(2), kernel_new.size(3)

    X_padding = _padding2d(X, padding, kernel.size(), padding_mode, constant_val, dilation, stride)    

    X_padding = X_padding.unfold(2, kernel_height, stride[0])
    X_padding = X_padding.unfold(3, kernel_width, stride[1])

    # Adjust the shape for grouped convolution
    X_padding = X_padding.view(batch_size, groups, in_channels // groups, X_padding.size(2), X_padding.size(3), kernel_height, kernel_width)
    kernel_new = kernel_new.view(groups, out_channels // groups, in_channels // groups, kernel_height, kernel_width)


    # Unfold the input and start to sum
    out = torch.einsum('bgihwkj,goikj->bgohw', X_padding, kernel_new)
    out = out.reshape(batch_size, out_channels, out.size(3), out.size(4)).contiguous()
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)

    return out

def signal_conv1d(X: torch.Tensor, kernel: torch.Tensor, bias: Optional[torch.Tensor] = None, stride: Union[int, list, tuple] = 1, padding: Union[int, list, tuple] = 0, padding_mode: str = 'zero', dilation: Union[int, list, tuple] = 1, constant_val: float = 0, is_correlation: bool = False, groups=1):
    '''
    DIY Conv1d
    ========================================

    Parameters
    ----------
    - X: The input of the convolution [batch_size, in_channels, seq_len]
    - kernel: The original conv kernel [out_channels, in_channels, kernel_len]
    - bias: The bias to be added upon the output.
    - stride: The stride, list or tuple or int.
    - padding: The padding, list or tuple or int. If stride is set as `'same'`, then the convolution will compute in same padding mode. Additionally, padding can also be set as `'valid'` or `'full'`.
    - dilation: The dilation, list or tuple or int.
    - constant_val: The value used to do constant padding.
    - is_correlation: If true, the function will execute correlation operation; otherwise it will execute convolution.

    Return
    ------
    - Output of the convolution.

    Attention
    ---------
    - All the computation will be based on float64.

    '''
    batch_size, in_channels, seq_len = X.size()
    assert in_channels == kernel.size(1) * groups, 'The input channels do not match the kernel channels times groups!'
    assert in_channels % groups == 0, 'The input channels must be divisible by groups!'

    if not (isinstance(stride, list) or isinstance(stride, tuple)):
        stride = (stride,)

    if not is_correlation:
        kernel = torch.flip(kernel, [2])
    
    if not (isinstance(dilation, list) or isinstance(dilation, tuple)):
        dilation = (dilation,)

    X_padding = _padding1d(X, padding, kernel.size(), padding_mode, constant_val, dilation, stride)    

    # Unfold the input and start to sum
    out = F.conv1d(X_padding, kernel, bias, stride, dilation=dilation, groups=groups)

    return out


def upsample(x:torch.Tensor, a, stride=2):
    # 创建一个新的数组，长度是原始数组的两倍减一
    new_length = stride * a
    new_arr = torch.zeros(size=(x.size(0), x.size(1), new_length), dtype=x.dtype, device=x.device)    
    # 将原始数组的值复制到新数组的奇数索引位置
    new_arr[:, :, ::stride] = x
    return new_arr

def downsample(x:torch.Tensor, stride=2):
    new_arr = x[:, :, ::stride]
    return new_arr

def decompose(x:torch.Tensor, h:torch.Tensor):
    basic_size = x.size(-1) - h.size(-1) + 1
    M = h.size(0)
    padding = math.ceil(basic_size / M) * M - basic_size
    x = padding1d(x, (padding, 0))
    coeffs = F.conv1d(x, h)
    coeffs = downsample(coeffs, M)
    return coeffs, math.ceil(basic_size / M), x

def reconstruct(coeffs:torch.Tensor, h:torch.Tensor, a):
    M = h.size(0)
    coeffs = upsample(coeffs, a, M)
    x = F.conv_transpose1d(coeffs, h)
    return x

def calculate_same_padding2d_np(height, width, kernel_size, stride, dilation):
    kernel_height, kernel_width = kernel_size[2], kernel_size[3]
    stride_height, stride_width = stride
    dilation_height, dilation_width = dilation
    
    # Calculate effective kernel size
    effective_kernel_height = (kernel_height - 1) * dilation_height + 1
    effective_kernel_width = (kernel_width - 1) * dilation_width + 1
    
    # Calculate total padding
    padding_height = max(0, (height - 1) * stride_height + effective_kernel_height - height)
    padding_width = max(0, (width - 1) * stride_width + effective_kernel_width - width)
    
    # Calculate padding for each side
    padding_top = padding_height // 2
    padding_bottom = padding_height - padding_top
    padding_left = padding_width // 2
    padding_right = padding_width - padding_left
    
    return (padding_top, padding_bottom, padding_left, padding_right)

def _padding2d_np(X, padding, kernel_size, padding_mode='zero', constant_val=0, dilation=(1,1), stride=(1,1)):
    batch_size, in_channels, height, width = X.shape
    kernel_height, kernel_width = kernel_size[2], kernel_size[3]
    
    if padding == 'same':
        padding = calculate_same_padding2d_np(height, width, kernel_size, stride, dilation)
    elif padding == 'valid':
        return X
    elif padding == 'full':
        padding = (kernel_height - 1, kernel_height - 1, kernel_width - 1, kernel_width - 1)
    else:
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        else:
            padding = (padding[0], padding[0], padding[1], padding[1])
    
    if padding_mode == 'zero' or padding_mode == 'constant':
        return np.pad(X, ((0, 0), (0, 0), (padding[0], padding[1]), (padding[2], padding[3])), mode='constant', constant_values=constant_val)
    elif padding_mode == 'wrap':
        return np.pad(X, ((0, 0), (0, 0), (padding[0], padding[1]), (padding[2], padding[3])), mode='wrap')
    elif padding_mode == 'reflect':
        return np.pad(X, ((0, 0), (0, 0), (padding[0], padding[1]), (padding[2], padding[3])), mode='reflect')
    elif padding_mode == 'copy':
        return np.pad(X, ((0, 0), (0, 0), (padding[0], padding[1]), (padding[2], padding[3])), mode='edge')
    else:
        raise ValueError("Invalid padding_mode!")

def conv2d_np(X, kernel, bias=None, stride=1, padding=0, padding_mode='zero', dilation=1, constant_val=0, is_correlation=True, groups=1):
    batch_size, in_channels, height, width = X.shape
    out_channels, kernel_in_channels, kernel_height, kernel_width = kernel.shape
    assert in_channels == kernel_in_channels * groups, 'The input channels does not match!'
    
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    
    if not is_correlation:
        kernel = np.flip(kernel, axis=(2, 3))
    
    # Expand the kernel
    if dilation[0] != 1 or dilation[1] != 1:
        kernel_new = expand_kernel_np(kernel, dilation)
    else:
        kernel_new = kernel
    kernel_height, kernel_width = kernel_new.shape[2], kernel_new.shape[3]
    
    # Padding
    X_padded = _padding2d_np(X, padding, kernel.shape, padding_mode, constant_val, dilation, stride)
    
    # Output dimensions
    out_height = (height + 2 * padding[0] - dilation[0] * (kernel_height - 1) - 1) // stride[0] + 1
    out_width = (width + 2 * padding[1] - dilation[1] * (kernel_width - 1) - 1) // stride[1] + 1
    
    # Unfold the input
    X_unfold = np.lib.stride_tricks.as_strided(
        X_padded,
        shape=(batch_size, in_channels, out_height, out_width, kernel_height, kernel_width),
        strides=(X_padded.strides[0], X_padded.strides[1], stride[0] * X_padded.strides[2], stride[1] * X_padded.strides[3], X_padded.strides[2], X_padded.strides[3])
    )
    
    # Adjust the shape for grouped convolution
    X_unfold = X_unfold.reshape(batch_size, groups, in_channels // groups, out_height, out_width, kernel_height, kernel_width)
    kernel_new = kernel_new.reshape(groups, out_channels // groups, in_channels // groups, kernel_height, kernel_width)
    
    # Perform convolution using einsum
    out = np.einsum('bgihwkj,goikj->bgohw', X_unfold, kernel_new)
    out = out.reshape(batch_size, out_channels, out_height, out_width)
    
    if bias is not None:
        out += bias.reshape(1, -1, 1, 1)
    
    return out

def expand_kernel_np(kernel, dilation):
    out_channels, in_channels, kernel_height, kernel_width = kernel.shape
    new_kernel_height = kernel_height + (kernel_height - 1) * (dilation[0] - 1)
    new_kernel_width = kernel_width + (kernel_width - 1) * (dilation[1] - 1)
    new_kernel = np.zeros((out_channels, in_channels, new_kernel_height, new_kernel_width), dtype=kernel.dtype)
    for i in range(kernel_height):
        for j in range(kernel_width):
            new_kernel[:, :, i * dilation[0], j * dilation[1]] = kernel[:, :, i, j]
    return new_kernel
