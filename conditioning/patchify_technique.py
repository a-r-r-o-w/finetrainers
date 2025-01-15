import torch

# single channel latent 4608x128 patchified
# doubled channel ... 
# torch.Size([1, 4608, 256]) 
# modify the linear layer ... (4608x256 and 128x2048) 

# Pipeline
# Video Input
# Video tensor torch.Size([1, 96, 3, 512, 768]) B F C H W
# to 
# Video tensor permuted torch.Size([1, 3, 96, 512, 768]) B C F H W

# Latent Encode temporal frames 8 compression 32 compression spatio
# channels go to 128
# torch.Size([1, 128, 12, 16, 24])

# Creating a Multi Channel Single Video
a = torch.randn(1,2,2,2) # video
print(a)

a = a.flatten(0,3)
print(a)
print(a.shape)

# b = torch.randn(1,2,4,4) # conditioning
# c = torch.cat([a,b],dim=1)

# Final shape.
# print(c.shape)
# print(c)


#_prepare_latents
# prepare_latents


# ImageOrVideoDataset Shape result 
# _preprocess_video [F, C, H, W] video tensor 4 

# _preprocess_image [1, C, H, W] (1-frame video)

# VAE Latent Shape Results Dims
# 4 second video 24 fps 
# torch.Size([1, 128, 12, 16, 24])


# VideoProcessor latent output shape preprocess_video

# Video Tensor Shape 

# Latent Tensor Shape 

# VAE Patchification Result Shape

# Unpacked latents of shape are [B, C, F, H, W] are patched into tokens of shape [B, C, F // p_t, p_t, H // p, p, W // p, p].
# The patch dimensions are then permuted and collapsed into the channel dimension of shape:
# [B, F // p_t * H // p * W // p, C * p_t * p * p] (an ndim=3 tensor).
# dim=0 is the batch size, dim=1 is the effective video sequence length, dim=2 is the effective number of input features

# encode and then decode and decouple the videos

# check transformer hidden state, should still be same sequence or slightly larger sequence.

def patchify_multichannel_latent(video_latent:torch.tensor,patch_size:int):# B, C, F, W, H 
    pass 

def unpatchify_multichannel_laten(video_latent:torch.tensor,patch_size:int): # B, C ,F, W, H
    pass

def mask_multichannel_latent(mask_shape:torch.tensor):
    pass