import cv2
import numpy as np
import torch
import math
import imgaug.augmenters as iaa
from einops import rearrange
import collections.abc as collections

def generate_target_foreground_mask(img, bg_threshold=10, bg_reverse= False):
        # convert RGB into GRAY scale
    try: img_gray    =   cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    except: img_gray    =   cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
    factor      =   0.1 
    for value in np.arange(0.1,1,0.1):
         if value*np.max(img_gray)>np.min(img_gray):
              factor    =   value
              break 
    try:
        bg_threshold                =   int(np.random.choice(np.arange(np.min(img_gray),np.max(img_gray)*factor),1)[0])
    except: bg_threshold            =   0
    # generate binary mask of gray scale image
    target_background_mask      = (img_gray<bg_threshold)
    target_background_mask      = target_background_mask.astype(np.bool_).astype(np.int64)

    # invert mask for foreground mask
    if bg_reverse:
        target_foreground_mask = target_background_mask
    else:
        target_foreground_mask = -(target_background_mask - 1)
    
    return target_foreground_mask

def lerp_np(x,y,w):
    fin_out = (y-x)*w + x
    return fin_out

def rand_perlin_2d_np(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    tt = np.repeat(np.repeat(gradients,d[0],axis=0),d[1],axis=1)

    tile_grads = lambda slice1, slice2: np.repeat(np.repeat(gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]],d[0],axis=0),d[1],axis=1)
    dot = lambda grad, shift: (
                np.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                            axis=-1) * grad[:shape[0], :shape[1]]).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * lerp_np(lerp_np(n00, n10, t[..., 0]), lerp_np(n01, n11, t[..., 0]), t[..., 1])

def generate_perlin_noise_mask(img, min_perlin_scale=0, perlin_scale=6, perlin_noise_threshold=0.5):
    # define perlin noise scale
    perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
    perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

    resized_img   = False
    # generate perlin noise
    try:
        perlin_noise = rand_perlin_2d_np((img.shape[0], img.shape[1]), (perlin_scalex, perlin_scaley))
    except:
        resized_img      = True
        orig_w,orig_h,_  = img.shape
        img              = cv2.resize(img,(256,256))
        perlin_noise     = rand_perlin_2d_np((img.shape[0], img.shape[1]), (perlin_scalex, perlin_scaley))
         
    # apply affine transform
    #rot             = iaa.Affine(rotate=(-90, 90))
    #perlin_noise    = rot(image=perlin_noise)
    
    # make a mask by applying threshold
    mask_noise = np.where(
        perlin_noise > perlin_noise_threshold, 
        np.ones_like(perlin_noise), 
        np.zeros_like(perlin_noise)
    )
    if resized_img:
        mask_noise = cv2.resize(mask_noise,(orig_w,orig_h))
    return mask_noise

def rand_augment():
        augmenters = [
            iaa.GammaContrast((0.5,2.0),per_channel=True),
            iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
            iaa.pillike.EnhanceSharpness(),
            iaa.AddToHueAndSaturation((-50,50),per_channel=True),
            iaa.Solarize(0.5, threshold=(32,128)),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize(),
            iaa.Affine(rotate=(-45, 45))
        ]

        aug_idx = np.random.choice(np.arange(len(augmenters)), 3, replace=False)
        aug = iaa.Sequential([
            augmenters[aug_idx[0]],
            augmenters[aug_idx[1]],
            augmenters[aug_idx[2]]
        ])
        
        return aug

def structure_source(img, structure_grid_size=8):
        structure_source_img = rand_augment()(image=img)
        
        assert img.shape[0] % structure_grid_size == 0, 'structure should be devided by grid size accurately'
        grid_w = img.shape[1] // structure_grid_size
        grid_h = img.shape[0] // structure_grid_size
        
        structure_source_img = rearrange(
            tensor  = structure_source_img, 
            pattern = '(h gh) (w gw) c -> (h w) gw gh c',
            gw      = grid_w, 
            gh      = grid_h
        )
        disordered_idx = np.arange(structure_source_img.shape[0])
        np.random.shuffle(disordered_idx)

        structure_source_img = rearrange(
            tensor  = structure_source_img[disordered_idx], 
            pattern = '(h w) gw gh c -> (h gh) (w gw) c',
            h       = structure_grid_size,
            w       = structure_grid_size
        ).astype(np.float32)
        
        return structure_source_img 

def anomaly_source(img, texture_img=""):
    if texture_img is not None: text_check = True
    else: text_check = False

    p = np.random.uniform() if text_check else 1.0
    if p < 0.5:
        anomaly_source_img = texture_img
    else:
        try:
            anomaly_source_img = structure_source(img)
        except:    
            orig_w, orig_h,_    = img.shape
            img                 = cv2.resize(img, dsize=(256, 256))
            anomaly_source_img  = structure_source(img)
            anomaly_source_img  = cv2.resize(img, dsize=(orig_h,orig_w))
            
    return anomaly_source_img

