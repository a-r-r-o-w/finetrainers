import torch

# Creating a Multi Channel Single Video
a = torch.ones(1,2,4,4) # video
b = torch.randn(1,2,4,4) # conditioning
c = torch.cat([a,b],dim=1)

# Final shape.
print(c.shape)
print(c)

# ImageOrVideoDataset Shape result 
# _preprocess_video [F, C, H, W] video tensor
# _preprocess_image [1, C, H, W] (1-frame video)

# VAE Shape Results Dims

# VideoProcessor latent output shape preprocess_video

# Video Tensor Shape 

# Latent Tensor Shape 

# VAE Patchification Result Shape

# encode and then decode and decouple the videos

#
# check transformer hidden state, should still be same sequence or slightly larger sequence.

def patchify_multichannel_latent(video_latent:torch.tensor,patch_size:int):# B, C, F, W, H 
    pass 

def unpatchify_multichannel_laten(video_latent:torch.tensor,patch_size:int): # B, C ,F, W, H
    pass

def mask_multichannel_latent(mask_shape:torch.tensor):
    pass