import random
import numpy as np 
from PIL import Image
import cv2

def crop_and_paste_(image, patch_w, patch_h, transform, rotation=False):
    """
    Crop patch from original image and paste it randomly on the same image.

    :image: [PIL] _ original image
    :patch_w: [int] _ width of the patch
    :patch_h: [int] _ height of the patch
    :transform: [binary] _ if True use Color Jitter augmentation
    :rotation: [binary[ _ if True randomly rotates image from (-45, 45) range

    :return: augmented image
    """

    org_w, org_h    = image.size
    mask = None
    
    try:
        patch_left, patch_top = random.randint(0, org_w - patch_w), random.randint(0, org_h - patch_h)
        patch_right, patch_bottom = patch_left + patch_w, patch_top + patch_h
        patch   = image.crop((patch_left, patch_top, patch_right, patch_bottom))
    except:
        image   =   Image.fromarray(np.asarray(image))
        patch   =   image.crop((patch_left, patch_top, patch_right, patch_bottom))
        
    while (np.min(patch.size)==0):
        patch_w, patch_h    =   np.random.choice(np.arange(1,int(0.25*org_w)),1)[0], np.random.choice(np.arange(1,int(0.25*org_h)),1)[0]
        patch_left, patch_top = random.randint(0, org_w - patch_w), random.randint(0, org_h - patch_h)
        patch_right, patch_bottom = patch_left + patch_w, patch_top + patch_h
        patch   =   image.crop((patch_left, patch_top, patch_right, patch_bottom))
        
    if transform:
        patch= transform(patch)

    if rotation:
        random_rotate   = random.uniform(*rotation)
        patch           = patch.convert("RGBA").rotate(random_rotate, expand=True)
        mask            = patch.split()[-1]

    paste_left, paste_top   = random.randint(0, org_w - patch_w), random.randint(0, org_h - patch_h)
    aug_image               = image.copy()
    aug_image.paste(patch, (paste_left, paste_top), mask=mask)
    mask_                   = Image.fromarray(np.zeros_like(aug_image))
    mask_.paste(patch, (paste_left, paste_top), mask=mask)
    img, label  =   np.asarray(aug_image), np.expand_dims(((np.asarray(mask_)>0)*255).astype(np.uint8)[:,:,0], axis=-1)
    
    return img, label

def patch_dim_extraction(image_np, area_ratio, aspect_ratio):
    img_area        = image_np.shape[0] * image_np.shape[1]
    patch_area      = random.uniform(*area_ratio) * img_area
    patch_aspect    = random.choice([random.uniform(*aspect_ratio[0]), random.uniform(*aspect_ratio[1])])
    patch_w         = int(np.sqrt(patch_area*patch_aspect))
    patch_h         = int(np.sqrt(patch_area/patch_aspect))
    return patch_w, patch_h