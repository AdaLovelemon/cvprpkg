from setuptools import setup, find_packages

setup(
    name='cvprpkg',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torchvision',
        'torch',
        'matplotlib',
        'Pillow'
    ],
    author='Ada Lovelemon',
    author_email='AdaLovelemon@gmail.com',
    description='This package is created to complete CVPR assignments.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)