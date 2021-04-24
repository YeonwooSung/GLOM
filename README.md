# GLOM

PyTorch implementation of [GLOM](https://arxiv.org/abs/2102.12627), Geoffrey Hinton's new idea that integrates concepts from neural fields, top-down-bottom-up processing, and attention (consensus between columns).

## 1. Overview

An implementation of Geoffrey Hinton's paper "How to represent part-whole hierarchies in a neural network" for MNIST Dataset.

## 2. Usage

### 2 - 1. PyTorch version

```python
import torch
from pyglom import GLOM

model = GLOM(
    dim = 512,         # dimension
    levels = 6,        # number of levels
    image_size = 224,  # image size
    patch_size = 14    # patch size
)

img = torch.randn(1, 3, 224, 224)
levels = model(img, iters = 12) # (1, 256, 6, 512) - (batch - patches - levels - dimension)
```

Pass the `return_all = True` keyword argument on forward, and you will be returned all the column and level states per iteration, (including the initial state, number of iterations + 1). You can then use this to attach any losses to any level outputs at any time step.

It also gives you access to all the level data across iterations for clustering, from which one can inspect for the theorized islands in the paper.

```python
import torch
from pyglom import GLOM

model = GLOM(
    dim = 512,         # dimension
    levels = 6,        # number of levels
    image_size = 224,  # image size
    patch_size = 14    # patch size
)

img = torch.randn(1, 3, 224, 224)
all_levels = model(img, iters = 12, return_all = True) # (13, 1, 256, 6, 512) - (time, batch, patches, levels, dimension)

# get the top level outputs after iteration 6
top_level_output = all_levels[7, :, :, -1] # (1, 256, 512) - (batch, patches, dimension)
```

Denoising self-supervised learning for encouraging emergence, as described by Hinton

```python
import torch
import torch.nn.functional as F
from torch import nn
from einops.layers.torch import Rearrange

from pyglom import GLOM

model = GLOM(
    dim = 512,         # dimension
    levels = 6,        # number of levels
    image_size = 224,  # image size
    patch_size = 14    # patch size
)

img = torch.randn(1, 3, 224, 224)
noised_img = img + torch.randn_like(img)

all_levels = model(noised_img, return_all = True)

patches_to_images = nn.Sequential(
    nn.Linear(512, 14 * 14 * 3),
    Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = 14, p2 = 14, h = (224 // 14))
)

top_level = all_levels[7, :, :, -1]  # get the top level embeddings after iteration 6
recon_img = patches_to_images(top_level)

# do self-supervised learning by denoising

loss = F.mse_loss(img, recon_img)
loss.backward()
```

You can pass in the state of the column and levels back into the model to continue where you left off (perhaps if you are processing consecutive frames of a slow video, as mentioned in the paper)

```python
import torch
from pyglom import GLOM

model = GLOM(
    dim = 512,
    levels = 6,
    image_size = 224,
    patch_size = 14
)

img1 = torch.randn(1, 3, 224, 224)
img2 = torch.randn(1, 3, 224, 224)
img3 = torch.randn(1, 3, 224, 224)

levels1 = model(img1, iters = 12)                   # image 1 for 12 iterations
levels2 = model(img2, levels = levels1, iters = 10) # image 2 for 10 iteratoins
levels3 = model(img3, levels = levels2, iters = 6)  # image 3 for 6 iterations
```

### 2 - 2. PyTorch-Lightning version

The pyglom also provides the GLOM model that is implemented with PyTorch-Lightning.

```python
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os
from pytorch_lightning.callbacks import ModelCheckpoint


from pyglom.glom import LightningGLOM


dataset = MNIST(os.getcwd(), download=True, transform=transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
]))
train, val = random_split(dataset, [55000, 5000])

glom = LightningGLOM(
    dim=256,         # dimension
    levels=6,        # number of levels
    image_size=256,  # image size
    patch_size=16,   # patch size
    img_channels=1
)

gpus = torch.cuda.device_count()
trainer = pl.Trainer(gpus=gpus, max_epochs=5)
trainer.fit(glom, DataLoader(train, batch_size=8, num_workers=2), DataLoader(val, batch_size=8, num_workers=2))
```

## 3. ToDo

- contrastive / consistency regularization of top-ish levels

## 4. Citations

```bibtex
@misc{hinton2021represent,
    title   = {How to represent part-whole hierarchies in a neural network}, 
    author  = {Geoffrey Hinton},
    year    = {2021},
    eprint  = {2102.12627},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```
