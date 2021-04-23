from setuptools import setup, find_packages

setup(
    name='pyglom',
    packages=find_packages(),
    version='0.0.1',
    license='MIT',
    description='Pytorch implementation of GLOM',
    author='Yeonwoo Sung',
    author_email='neos960518@gmail.com',
    url='https://github.com/YeonwooSung/GLOM',
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
        'torch>=1.6'
    ],
    classifiers=[
        'Development Status :: 1 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
)
