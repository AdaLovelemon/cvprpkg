import numpy as np
import torch

from typing import Union

from .kernel import GaussianKernel2d
from .ConvPkg import conv2d

def bilinear_interpolate(image_original:Union[torch.Tensor, np.ndarray], x:Union[torch.Tensor, np.ndarray], y:Union[torch.Tensor, np.ndarray]):
    if isinstance(image_original, np.ndarray):
        # x轴在后，y轴在前
        x0 = np.floor(x).astype(int)
        x1 = x0 + 1
        y0 = np.floor(y).astype(int)
        y1 = y0 + 1
        
        # 原地操作
        np.clip(x0, 0, image_original.shape[1] - 1, out=x0)
        np.clip(x1, 0, image_original.shape[1] - 1, out=x1)
        np.clip(y0, 0, image_original.shape[0] - 1, out=y0)
        np.clip(y1, 0, image_original.shape[0] - 1, out=y1)

    elif isinstance(image_original, torch.Tensor):
        x0 = torch.floor(x).to(int)
        x1 = x0 + 1
        y0 = torch.floor(y).to(int)
        y1 = y0 + 1
        
        # 原地操作
        torch.clip(x0, 0, image_original.shape[1] - 1, out=x0)
        torch.clip(x1, 0, image_original.shape[1] - 1, out=x1)
        torch.clip(y0, 0, image_original.shape[0] - 1, out=y0)
        torch.clip(y1, 0, image_original.shape[0] - 1, out=y1)

    if image_original.ndim == 2:
        Ia = image_original[y1, x1]
        Ib = image_original[y0, x1]
        Ic = image_original[y0, x0]
        Id = image_original[y1, x0]

    elif image_original.ndim == 3:
        Ia = image_original[:, y1, x1]
        Ib = image_original[:, y0, x1]
        Ic = image_original[:, y0, x0]
        Id = image_original[:, y1, x0]
    
    else:
        Ia = image_original[:, :, y1, x1]
        Ib = image_original[:, :, y0, x1]
        Ic = image_original[:, :, y0, x0]
        Id = image_original[:, :, y1, x0]

    wa = (x - x0) * (y - y0)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y1 - y)
    wd = (x1 - x) * (y - y0)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id

def _bilinear_interpolate(image, coords, weights):
    wa, wb, wc, wd = weights
    x0, x1, y0, y1 = coords

    Ia = image[y1, x1]
    Ib = image[y0, x1]
    Ic = image[y0, x0]
    Id = image[y1, x0]
    return wa * Ia + wb * Ib + wc * Ic + wd * Id

def nearest_neighbor_interpolate(image_original: Union[torch.Tensor, np.ndarray], x: Union[torch.Tensor, np.ndarray], y: Union[torch.Tensor, np.ndarray]):
    if isinstance(image_original, np.ndarray):
        # Nearest neighbor coordinates
        x_nearest = np.round(x).astype(int)
        y_nearest = np.round(y).astype(int)
        
        # Clip coordinates to be within image boundaries
        np.clip(x_nearest, 0, image_original.shape[1] - 1, out=x_nearest)
        np.clip(y_nearest, 0, image_original.shape[0] - 1, out=y_nearest)
        
    elif isinstance(image_original, torch.Tensor):
        # Nearest neighbor coordinates
        x_nearest = torch.round(x).to(int)
        y_nearest = torch.round(y).to(int)
        
        # Clip coordinates to be within image boundaries
        torch.clip(x_nearest, 0, image_original.shape[1] - 1, out=x_nearest)
        torch.clip(y_nearest, 0, image_original.shape[0] - 1, out=y_nearest)

    if image_original.ndim == 2:
        return image_original[y_nearest, x_nearest]
    elif image_original.ndim == 3:
        return image_original[:, y_nearest, x_nearest]
    else:
        return image_original[:, :, y_nearest, x_nearest]

def geometric_transform_bi_fast(image, transform_inv_func):
    op = np if isinstance(image, np.ndarray) else torch
    height, width = image.shape[-2], image.shape[-1]
    coords = np.indices((height, width)).reshape(2, -1)
    # [x, y, 1]
    coords = np.vstack((coords, np.ones((1, coords.shape[1]))))
    if op == torch:
        coords = torch.from_numpy(coords)

    new_coords = transform_inv_func(coords)
    new_coords = new_coords[:2].reshape(2, height, width)

    output_image = op.zeros_like(image)
    
    # Compute (x0, x1, y0, y1) and (wa, wb, wc, wd)
    x0 = op.floor(new_coords[1]).astype(int)
    x1 = x0 + 1
    y0 = op.floor(new_coords[0]).astype(int)
    y1 = y0 + 1

    op.clip(x0, 0, image.shape[-1] - 1, out=x0)
    op.clip(x1, 0, image.shape[-1] - 1, out=x1)
    op.clip(y0, 0, image.shape[-2] - 1, out=y0)
    op.clip(y1, 0, image.shape[-2] - 1, out=y1)
    wa = (new_coords[1] - x0) * (new_coords[0] - y0)
    wb = (new_coords[1] - x0) * (y1 - new_coords[0])
    wc = (x1 - new_coords[1]) * (y1 - new_coords[0])
    wd = (x1 - new_coords[1]) * (new_coords[0] - y0)

    # Multi-Channels
    for i in range(image.shape[1]):
        output_image[:, i] = _bilinear_interpolate(image[:, i], (x0, x1, y0, y1), (wa, wb, wc, wd))
    return output_image

def geometric_transform(image, transform_inv_func, order=1):
    '''
    Templates
    =========
    
    >>> def transform(coords):
             new_coords = M @ coords
             return new_coords
    >>> geometric_transform(image, transform)

    or

    >>> M_inv = np.array([
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]
        ])
        geometric_transform(image, M_inv)
    
    '''
    height, width = image.shape[-2], image.shape[-1]
    op = np if isinstance(image, np.ndarray) else torch
    coords = np.indices((height, width)).reshape(2, -1)
    # [x, y, 1]
    coords = np.vstack((coords, np.ones((1, coords.shape[1]))))
    if op == torch:
        coords = torch.from_numpy(coords).float()

    if callable(transform_inv_func):
        new_coords = transform_inv_func(coords)
    elif isinstance(transform_inv_func, np.ndarray):
        if op == np: new_coords = transform_inv_func @ coords
        else: 
            transform_inv_func = torch.from_numpy(transform_inv_func)
            new_coords = torch.matmul(transform_inv_func, coords)
    elif isinstance(transform_inv_func, torch.Tensor):
        if op == torch: new_coords = torch.matmul(transform_inv_func, coords)
        else:
            transform_inv_func = transform_inv_func.numpy()
            new_coords = transform_inv_func @ coords
    else:
        raise TypeError('transform_inv_func should be a function or a Matrix')
    new_coords = new_coords[:2].reshape(2, height, width)

    output_image = op.zeros_like(image)
    
    # Multi-Channels
    if order == 0:
        interpolate = nearest_neighbor_interpolate
    elif order == 1:
        interpolate = bilinear_interpolate

    for i in range(image.shape[1]):
        output_image[:, i] = interpolate(image[:, i].reshape(image.shape[-2], image.shape[-1]), new_coords[1], new_coords[0])

    return output_image

# All following functions are inversed transforms
def Translation(tx, ty, numpy_array=False):
    T = np.array([
        [1, 0, -tx],
        [0, 1, -ty],
        [0, 0, 1]
    ])
    return T if numpy_array else torch.from_numpy(T).float()

def Rotation(axis_vec, theta, numpy_array=False):
    if isinstance(axis_vec, torch.Tensor):
        axis_vec = axis_vec.numpy()
    if isinstance(axis_vec, (list, tuple)):
        axis_vec = np.array(axis_vec)
    axis_vec = axis_vec.reshape(-1, 1)
    norm = np.linalg.norm(axis_vec)
    assert norm > 0, 'norm equals to 0'
    axis_vec = axis_vec / norm
    
    axis_vec_mat = np.array([
        [0, axis_vec[2, 0], -axis_vec[1, 0]],
        [-axis_vec[2, 0], 0, axis_vec[0, 0]],
        [axis_vec[1, 0], -axis_vec[0, 0], 0]
    ])
    cos_theta = np.cos(theta)
    R = cos_theta * np.eye(3) + (1 - cos_theta) * np.dot(axis_vec, axis_vec.T) + axis_vec_mat * np.sin(theta)
    return R if numpy_array else torch.from_numpy(R).float()

def Euclidean(theta, tx, ty, numpy_array=False):
    E = np.array([
        [np.cos(theta), np.sin(theta), -tx],
        [-np.sin(theta), np.cos(theta), -ty],
        [0, 0, 1]
    ])
    return E if numpy_array else torch.from_numpy(E).float()

def Similarity(scaling, theta, tx, ty, numpy_array=False):
    S = np.array([
        [scaling * np.cos(theta), np.sin(theta), -tx],
        [-np.sin(theta), scaling * np.cos(theta), -ty],
        [0, 0, 1]
    ])
    return S if numpy_array else torch.from_numpy(S).float()

def DownSampling(image:Union[torch.Tensor, np.ndarray], DownFactor:Union[int, tuple, list]=2):
    if not isinstance(DownFactor, (list, tuple)):
        DownFactor = (DownFactor, DownFactor)
    if isinstance(image, torch.Tensor):
        image_new = image.clone()[:, :, ::DownFactor[0], ::DownFactor[1]]
    
    elif isinstance(image, np.ndarray):
        if image.ndim == 3:
            image_new = image[:, ::DownFactor[0], ::DownFactor[1]]
        elif image.ndim == 2:
            image_new = image[::DownFactor[0], ::DownFactor[1]]
        else:
            raise ValueError("Invalid image shape!")
    else:
        raise TypeError("Image should be numpy.ndarray or torch.Tensor")

    return image_new

def UpSampling(image:Union[torch.Tensor, np.ndarray], UpFactor:Union[int, tuple, list]=2):
    if not isinstance(UpFactor, (list, tuple)):
        UpFactor = (UpFactor, UpFactor)
    if isinstance(image, torch.Tensor):
        batch_size, num_channels, height, width = image.size()
        image_new = torch.zeros(size=(batch_size, num_channels, height * UpFactor[0], width * UpFactor[1]))
        image_new[:, :, ::UpFactor[0], ::UpFactor[1]] = image

    elif isinstance(image, np.ndarray):
        if image.ndim == 3:
            num_channels, height, width = image.shape
            image_new = np.zeros((num_channels, height * UpFactor[0], width * UpFactor[1]))
            image_new[:, ::UpFactor[0], ::UpFactor[1]] = image
        elif image.ndim == 2:
            height, width = image.shape
            image_new = np.zeros((height * UpFactor[0], width * UpFactor[1]))
            image_new[::UpFactor[0], ::UpFactor[1]] = image
        else:
            raise ValueError("Invalid image shape!")
        
    else:
        raise TypeError("Image should be numpy.ndarray or torch.Tensor")

    return image_new

def GaussianPyramid(image, Factor, levels, kernel_size=None, sigma=None):  
    if kernel_size is None:
        kernel_size = 5
    if sigma is None:
        sigma = Factor * 0.5

    G_Pyramid, L_Pyramid = [image], []

    # 多通道滤波，这里的核心其实是取groups=3（分通道卷积），其次要保证卷积核是充分归一化了的(c, h, w)维归一化
    G = GaussianKernel2d(kernel_size, sigma, 1, image.shape[1])
    for i in range(levels):
        # Filtering before DownSampling to reduce high frequencies
        image_blurred = conv2d(image, G, padding='same', padding_mode='wrap', groups=3)
        image_detail = image - image_blurred
        L_Pyramid.append(image_detail)
        image = DownSampling(image_blurred, Factor)
        G_Pyramid.append(image)
    
    return G_Pyramid, L_Pyramid