from math import sqrt
import torch
import torch.nn.functional as F
from torch import nn, einsum
import pytorch_lightning as pl

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from glom.glom import GLOM



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
        lr=1e-3
    ):
        super().__init__()        
        self.lr = lr

        # a GLOM model
        self.glom = GLOM(
            dim=dim, 
            levels=levels,
            image_size=image_size, 
            patch_size=patch_size, 
            consensus_self=consensus_self, 
            local_consensus_radius=local_consensus_radius
        )

        # use MSE loss as a loss function
        self.loss_func = F.mse_loss

        # a network that converts the generated patches to images
        self.patches_to_images = nn.Sequential(
            nn.Linear(dim, patch_size * patch_size * 3),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = 14, p2 = 14, h = (224 // 14))
        )

    
    def forward(self, img, iters=None, levels=None, return_all=False):
        levels = self.glom(img, iters=iters, levels=levels, return_all=return_all)
        return levels


    def calculate_loss(self, img):
        # add random noise to images
        noised_img = img + torch.randn_like(img)

        # forward propagation
        all_levels = self(noised_img, return_all=True)

        #TODO
        # reconstruct images from patches
        recon_img = None

        # calculate loss
        loss = self.loss_func(img, recon_img)

        return loss


    def training_step(self, batch, batch_idx):
        imgs, y = batch

        if torch.cuda.is_available():
            loss = Variable(torch.zeros(1).cuda(), requires_grad=True)
        else:
            loss = Variable(torch.zeros(1), requires_grad=True)

        # iterate all images in the mini batch
        for img in imgs:
            l = calculate_loss(img)
            loss += l
        tensorboard_logs = {'train_loss': loss}

        return {'loss': loss, 'log': tensorboard_logs}


    def validation_step(self, batch, batch_nb):
        imgs, y = batch

        if torch.cuda.is_available():
            loss = Variable(torch.zeros(1).cuda(), requires_grad=True)
        else:
            loss = Variable(torch.zeros(1), requires_grad=True)

        # iterate all images in the mini batch
        for img in imgs:
            l = calculate_loss(img)
            loss += l
        return {'val_loss': loss}


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
