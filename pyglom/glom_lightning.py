from math import sqrt
import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.autograd import Variable
import pytorch_lightning as pl

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from pyglom.glom import GLOM



#------------------------------
# main class
#------------------------------

class LightningGLOM(pl.LightningModule):
    def __init__(
        self,
        *,
        dim=512,
        levels=6,
        image_size=224,
        patch_size=14,
        consensus_self=False,
        local_consensus_radius=0,
        lr=1e-3,
        img_channels=3
    ):
        super().__init__()        
        self.lr, self.levels = lr, levels

        # a GLOM model
        self.glom = GLOM(
            dim=dim, 
            levels=self.levels,
            image_size=image_size, 
            patch_size=patch_size, 
            consensus_self=consensus_self, 
            local_consensus_radius=local_consensus_radius,
            img_channels=img_channels
        )

        # use MSE loss as a loss function
        self.loss_func = F.mse_loss

        # a network that converts the generated patches to images
        self.patches_to_images = nn.Sequential(
            nn.Linear(dim, patch_size ** 2 * img_channels),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = 14, p2 = 14, h = (224 // 14))
        )

    
    def forward(self, img, iters=None, levels=None, return_all=False):
        all_levels = self.glom(img, iters=iters, levels=levels, return_all=return_all)
        return all_levels


    def calculate_loss(self, img):
        # change the shape of the tensor from 3D to 4D
        img = img.unsqueeze(0)
        # add random noise to images
        noised_img = img + torch.randn_like(img)

        # forward propagation
        all_levels = self(noised_img, return_all=True)


        # Reconstruct images from patches.

        # Get the top level embeddings after iteration n, where n is (# of levels + 1)
        # This is because the GLOM model needs to have twice the number of levels of iterations
        # in order for information to propagate up and back down.
        recon_img = all_levels[self.levels + 1, :, :, -1]

        # calculate loss
        loss = self.loss_func(img, recon_img.unsqueeze(0))

        return loss


    def training_step(self, batch, batch_idx):
        imgs, y = batch
        loss = None

        # iterate all images in the mini batch
        for img in imgs:
            if loss:
                loss += self.calculate_loss(img)
            else:
                loss = self.calculate_loss(img)
        tensorboard_logs = {'train_loss': loss}

        return {'loss': loss, 'log': tensorboard_logs}


    def validation_step(self, batch, batch_nb):
        imgs, y = batch
        loss = None

        # iterate all images in the mini batch
        for img in imgs:
            if loss:
                loss += self.calculate_loss(img)
            else:
                loss = self.calculate_loss(img)
        tensorboard_logs = {'val_loss': loss}

        return {'loss': loss, 'val_loss': loss ,'log': tensorboard_logs}


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
