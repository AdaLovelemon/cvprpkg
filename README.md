# cvprpkg
This package, named `cvprpkg`, was created to complete assignments at Xi'an Jiaotong University.

## Features

- ConvPkg: Basic Convolutions of images
- kernel: GaussianKernel and other kernels
- Transforms: Basic Parameterized Geometric Transforms
- image_utils: Read and Display Images
- EdgeCornerDetection: Canny Edge Detection & Harris Corner Detection (Waited to be completed)
- Fourier: Fourier Transform of an image

## Installation

To install this package from GitHub, use the following command:

```bash
pip install git+https://github.com/AdaLovelemon/cvprpkg.git
```

## Usage

Here is an example of how to use this package:

```python
from cvprpkg.ConvPkg import conv2d
from cvprpkg.kernel import GaussianKernel2d
from cvprpkg.image_utils import read_img, imshow

# Basically, all images involved need to be in this form: [batch_size, num_channels, height, width]
img = read_img('Your/image/path.jpg', True, False)
feature_map = conv2d(img, GaussianKernel2d(), padding='same')
imshow(feature_map)
```

## Acknowledgement

This package was developed as part of university assignments at [IAIR-XJTU](http://www.aiar.xjtu.edu.cn/). It may contain bugs or incomplete features.

## Contributions

Contributions are welcome! If you find any bugs or have ideas for improvements, feel free to open an issue or submit a pull request on GitHub. Your contributions will help make this package better for everyone.

## Author

**Ada Lovelemon** - Initial work - [AdaLovelemon@gmail.com](mailto:AdaLovelemon@gmail.com)

## License

This project is licensed under the MIT License.

## Contact

For any questions or issues, please contact **Ada Lovelemon** at [AdaLovelemon@gmail.com](mailto:AdaLovelemon@gmail.com).
