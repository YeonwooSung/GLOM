from setuptools import setup, find_packages
import os.path as path

basedir = path.abspath(path.dirname(__file__))
with open(path.join(basedir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='pyglom',
    packages=find_packages(),
    version='0.0.2',
    license='MIT',
    description='PyTorch implementation of GLOM. PyTorch-Lightning version is also available.',
    author='Yeonwoo Sung',
    author_email='neos960518@gmail.com',
    url='https://github.com/YeonwooSung/GLOM',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=[
        'AI',
        'artificial intelligence',
        'deep learning',
        'glom',
        'neuralnet',
        'neural network',
        'NN'
    ],
    install_requires=[
        'einops>=0.3',
        "pytorch-lightning==0.8.5",
        'torch>=1.6'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6'
    ],
)
