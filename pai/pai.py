import random
import numpy as np
from torchvision import transforms
import cv2
from PIL import Image
import torch
from skimage.segmentation import slic
from scipy import ndimage
from skimage.util import random_noise
import os
import warnings
import io
from opensimplex import noise3array as noise3array_fast
from scipy.ndimage import rotate
from time import time
############################################################

########################################################
from perlin import *
from mem_seg_perlin import *
from rand_aug_cut_paste import *
from simplex_noise import *
from random_pert import *
from hard_augment_cutpaste import * 
from fractal_aug import *
from cutpaste import *
from augmented_anomlay import *
from change_color import *
########################################################
class Anomaly_Insertion(object):

    def __init__(self, transform = True):

        '''
        This class creates to different augmentation to Pseudo-Anomaly Insertion Methods. 
        :arg
        :transform[binary]: - if True use Color Jitter augmentations for patches
        '''
        if transform:
            self.transform      = transforms.ColorJitter(brightness = 0.1,
                                                      contrast = 0.1,
                                                      saturation = 0.1,
                                                      hue = 0.1)
        else:
            self.transform      = None
        self.transparency_range = [0.15, 1.]
    
    def cutpaste_scar(self, image, rotation = (-45, 45)):
        '''
        Generate a small fragment from a normal image to act as an anomaly 
        and randomly paste it back into the normal image
        
        Args:
            image (PIL.Image, mode=RGB) or numpy image: The original image
            rotation: [tuple] - range for rotation
        Returns:
            numpy.ndarray: (anomalous_image,mask): The image with the anomaly inserted and segmentation mask 
        '''
        orig_img    =   np.asarray(image)   
        image       =   Image.fromarray(np.asarray(image))
        mask        =   np.zeros_like(np.asarray(image))
        count       =   0
        while (np.sum(mask)==0) and count<=5:
            img_w, img_h        = image.size
            w_r                 = np.random.choice(np.arange(1,3),1)[0]
            width               = [int(img_w*0.1), int(img_w*0.20*w_r)]
            #############################################################    
            if np.min(width)==0:
                width           = [1, int(img_w*0.8)]
            #############################################################    
            w_l                 = np.random.choice(np.arange(1,3),1)[0]
            height              = [int(img_h*0.1*w_l), int(img_h*0.20*w_l)]
            if np.min(height)==0:
                width           = [1, int(img_h*0.8)]
            #############################################################    
            
            if img_h<np.max(height):
                height          = [1,img_h-1]
            if img_w<np.max(width):
                width           = [1,img_w-1]
            patch_w, patch_h    = random.randint(*width), random.randint(*height)
            if patch_w==0: patch_w              = 1
            if patch_h==0: patch_h              = 1
            if patch_w>=img_w:
                pass
            if patch_h>=img_h:
                pass
            aug_img, mask       = crop_and_paste_(image, patch_w, patch_h, self.transform, rotation = rotation)
            count+=1
        if np.sum(mask)==0: 
            print("Anomaly Generation Fails: Kindly Rerun this method")
        return aug_img, mask 

    def perlin_noise_pattern(self, image, anom_source_img=None,resize=None):
        """
        Insert an anomaly into the image by generating perlin noise patterns and cropped out the pattern from anomaly soutce image to paste into normal image
        
        Args:
            image (PIL.Image, mode=RGB) or numpy image: The original image.
            anom_source_img (PIL.Image, mode=RGB): The anomaly source image utilize to be insert anomaly.
        Returns:
            numpy.ndarray: (anomalous_image,mask): The image with the anomaly inserted and segmentation mask
        """
        try: orig_h,orig_w = image.size
        except: orig_h,orig_w,_ = image.shape
        orig_img        =   np.copy(image)
        image           =   cv2.resize(np.asarray(image),(256,256))
        anom_source_img =   cv2.resize(np.asarray(anom_source_img),(256,256))
        msk                 = np.zeros_like(image)
        count = 0
        
        while np.sum(msk)==0 and count<=5:
            perlin_scale        = 6
            min_perlin_scale    = 0
            if resize is not None:
                msk                 = np.ones((resize(0),resize(1)))
            else:
                try:
                    msk                 = np.ones((image.size[0],image.size[1]))
                except:
                    msk                 = np.ones((image.shape[0],image.shape[1]))
                    
            perlin_scalex   = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
            perlin_scaley   = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
            resize_img      = False
            #while np.count_nonzero(msk)>(0.1*msk.shape[0]*msk.shape[1]):
            try:
                try: perlin_noise        = rand_perlin_2d_np((image.shape[0], image.shape[1]), (perlin_scalex, perlin_scaley))
                except:
                    image               =   np.asarray(image)
                    perlin_noise        =   rand_perlin_2d_np((image.shape[0], image.shape[1]), (perlin_scalex, perlin_scaley))
            except:
                    resize_img          =   True          
                    image               =   cv2.resize(np.asarray(image),(256,256)) 
                    anom_source_img     =   cv2.resize(np.asarray(anom_source_img),(256,256))   
                    perlin_noise        =   rand_perlin_2d_np((image.shape[0], image.shape[1]), (perlin_scalex, perlin_scaley))  
            perlin_thr          = np.where(perlin_noise >  np.random.uniform(low=np.min(perlin_noise), high=np.max(perlin_noise),size=1)[0], np.ones_like(perlin_noise), np.zeros_like(perlin_noise))

            perlin_thr          = np.expand_dims(perlin_thr, axis=2)
            msk                 = (np.copy(perlin_thr)*255).astype(np.uint8)
            count+=1
          
        image               = np.asarray(image)/255
        if anom_source_img is not None: image_  = np.asarray(anom_source_img)
        else: image_    =   np.copy(image)
        img_thr             = (image_.astype(np.float32) * perlin_thr) / 255.0
        beta                = torch.rand(1).numpy()[0] * 0.8
        augmented_image     = ((image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (perlin_thr))*255).astype(np.uint8)
        
        if resize_img:
            augmented_image               =   cv2.resize(np.asarray(augmented_image),(orig_h,orig_w))    
            msk                           =   cv2.resize(np.asarray(msk),(orig_h,orig_w))
        aug_img, mask   =   cv2.resize(augmented_image,(orig_h,orig_w)), cv2.resize(msk,(orig_h,orig_w))
        aug_img         = (((1-np.expand_dims(mask/255,axis=-1))*orig_img) + (np.expand_dims(mask,axis=-1)/255)*aug_img).astype(np.uint8)
        if np.sum(msk)==0: 
            print("Not able to add anomaly into normal image:Change the anomaly source image")
        return augmented_image, msk
    
    def superpixel_anomaly(self, image, anom_source_img=None):
        '''
        Insert an anomaly into the image by generating superpixel patterns of different size from anomaly source image 
        
        Args:
            image (PIL.Image, mode=RGB) or numpy image: The original image.
            anom_source_img (PIL.Image, mode=RGB): The anomaly source image utilize to be insert anomaly.
        Returns:
            numpy.ndarray: (anomalous_image,mask): The image with the anomaly inserted and segmentation mask. 
        '''
        image               =  np.asarray(image) 
        if anom_source_img is None: anom_source_img   =   np.copy(image)
        else: anom_source_img    =   np.asarray(anom_source_img)
        
        seg_choice          = [30, 150, 250] #[10,20,30,40,50,60,70,150,200,250] # [150,200,250] #[10,20,30,40,50,60,70] #[5,10,20,30,40,50]
        choice_for_seg_anom = np.random.choice(seg_choice,len(seg_choice))[0]
        
        choice_for_rotate       = np.random.choice(3,3)[0]
        anomaly_img_augmented_orignal   =   np.copy(anom_source_img)
        if choice_for_rotate==0:
            anomaly_img_augmented    = cv2.rotate(anom_source_img, cv2.ROTATE_90_CLOCKWISE)
        elif choice_for_rotate==1:
            anomaly_img_augmented    = cv2.rotate(anom_source_img, cv2.ROTATE_180)
        else:
            anomaly_img_augmented    =  np.copy(anom_source_img)
        #t1                      =       timeit.default_timer()
        segments_slic           =       slic(anomaly_img_augmented.reshape((image.shape[0],image.shape[1],image.shape[2])), n_segments=choice_for_seg_anom, compactness=8, sigma=1,start_label=1)
        no_of_seg               =       np.unique(segments_slic)
        uniq_v                  =       np.random.choice(no_of_seg, int(np.ceil(len(no_of_seg)*0.05))) # uniq_v = np.random.choice(no_of_seg)   
        seg_img                 =       np.isin(segments_slic,uniq_v)

        msk                     =       np.expand_dims(seg_img, axis=2)
        try:
            augmented_image         =       msk * anomaly_img_augmented + (1-msk)*image
        except:
            augmented_image         =       msk * anomaly_img_augmented_orignal + (1-msk)*image
            
        msk                     =       (msk*255).astype(np.uint8)
        if np.sum(msk)==0: 
            print("Anomaly Generation Fails: Kindly Rerun this method")
        return augmented_image.astype(np.uint8), msk
    
    def perlin_with_roi_anomaly(self, img, anom_source_img=None, use_mask= True, transparency_range=[0.15, 1.]):
        '''
        Insert an anomaly into the image by generating perlin noise patterns and cropped out the pattern from anomaly soutce image to paste into normal image using the Region of Interest either provide
        by the user or compute automatically
         
        Args:
            image (PIL.Image, mode=RGB) or numpy image: The original image.
            anom_source_img (PIL.Image, mode=RGB): The anomaly source image utilize to be insert anomaly.
            use_mask (boolean) : by default (True), if false provided by user so it generated results equivalnet to 
            perlin noise patterns
        Returns:
            numpy.ndarray: (anomalous_image,mask): The image with the anomaly inserted and segmentation mask. 
        '''
        try: orig_h,orig_w = img.size
        except: orig_h,orig_w,_ = img.shape
        orig_img        =   np.copy(img)
        img             =   cv2.resize(np.asarray(img),(256,256))
        anom_source_img =   cv2.resize(np.asarray(anom_source_img),(256,256))
        mask    =   np.zeros_like(np.asarray(img))
        _check  =   0
        while (np.sum(mask)==0) and _check<=10:
            # step 1. generate mask
            img                         = np.asarray(img)  
            ## target foreground mask
            if use_mask:
                
                target_foreground_mask  = generate_target_foreground_mask(img=img)
            else:
                target_foreground_mask  = np.ones((img.shape[1], img.shape[0]))
            
            ## perlin noise mask
            perlin_noise_mask           = generate_perlin_noise_mask(img)
            
            ## mask
            w, h                =   target_foreground_mask.shape
            mask                =   cv2.resize(perlin_noise_mask,(h,w)) * target_foreground_mask
            mask_expanded       =   np.expand_dims(mask, axis=2)
            
            # step 2. generate texture or structure anomaly
            if anom_source_img is not None: anom_sorce_img   = np.asarray(anom_source_img)
            ## anomaly source
            anomaly_source_img  = anomaly_source(img=img, texture_img=anom_sorce_img)
            
            ## mask anomaly parts
            factor              = np.random.uniform(transparency_range)[0]
            anomaly_source_img  = factor * (mask_expanded.reshape(anomaly_source_img.shape[0],anomaly_source_img.shape[1],1) * anomaly_source_img) + (1 - factor) * (mask_expanded.reshape(anomaly_source_img.shape[0],anomaly_source_img.shape[1],1) * img)
            
            # step 3. blending image and anomaly source
            aug_img             = (((- mask_expanded + 1) * img) + anomaly_source_img).astype(np.uint8)
            mask                = np.expand_dims((mask*255).astype(np.uint8),axis=-1)
            _check+=1
            if _check>=5: use_mask = False 
        aug_img, mask   =   cv2.resize(aug_img,(orig_h,orig_w)), cv2.resize(mask,(orig_h,orig_w))
        if np.sum(mask)==0: 
            print("Not able to add anomaly into normal image:Change the anomaly source image")
        aug_img         = (((1-np.expand_dims(mask/255,axis=-1))*orig_img) + (np.expand_dims(mask,axis=-1)/255)*aug_img).astype(np.uint8)
        return aug_img, mask
    
    def rand_augmented_cut_paste(self, img_norm, anom_source_img=None, mask=None, in_fg_region=True):
        '''
        Insert an anomaly into the image by randomly augmented patterns from anomaly source image 
        to paste into normal image
        
        Args:
            image (PIL.Image, mode=RGB) or numpy image: The original image.
            anom_source_img (PIL.Image, mode=RGB): The anomaly source image utilize to be insert anomaly.
            mask (PIL or numpy image) : by default (None), if provided by user so it add anomaly inside ROI
            in_fg_region (boolean): by default (True), if false provided by user so it neglects the mask
        Returns:
            numpy.ndarray: (anomalous_image,mask): The image with the anomaly inserted and segmentation mask. 
        '''
        aug_mask        =    np.zeros_like(np.asarray(img_norm))
        repeat          =   0
        while np.sum(aug_mask)==0 or repeat<=5:
            repeat+=1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                aug     = randAugmenter()
            
            n_image = np.asarray(img_norm).copy()  # normal sample

            if anom_source_img is None: 
                image      = np.asarray(img_norm)  # anomaly sample
            else: image      = np.asarray(anom_source_img)

            if mask is None: 
                mask           = (np.ones((image.shape[0],image.shape[1]))*255).astype(np.uint8)
                rand_zeros     = np.random.randint(image.shape[0])   
                mask[:rand_zeros,:rand_zeros]           =  0

            img_height, img_width = n_image.shape[0], n_image.shape[1]

            
            mask = np.asarray(mask).copy()  # (900, 900)
            
            # augmente the abnormal region
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                augmentated         = aug(image=image, mask=mask)
            aug_image, aug_mask = augmentated['image'], augmentated['mask']
            
            perlin_noise_mask           = generate_perlin_noise_mask(aug_image)
            h,w                         = perlin_noise_mask.shape
            aug_image, aug_mask         = cv2.resize(aug_image,(w,h)), cv2.resize(aug_mask,(w,h))       
                ## mask
            aug_image                =   (np.expand_dims(perlin_noise_mask,axis=-1) * aug_image).astype(np.uint8)
            aug_mask                 =   (perlin_noise_mask*aug_mask).astype(np.uint8)
            # temp_img = Image.fromarray(aug_image)
            # temp_img.save("aug_imgs/ano_aug_img.jpg")
            # crop_img = aug_image.copy()
            # crop_img[aug_mask == 0] = 0
            # crop_img = Image.fromarray(crop_img)
            # crop_img.save('aug_imgs/crop_img.jpg')
            if in_fg_region:
                fg_mask = np.asarray(mask).copy()
                try:
                    intersect_mask = np.logical_and(fg_mask == 255, aug_mask == 255)
                except:
                    aug_mask       = aug_mask.reshape(fg_mask.shape[0],fg_mask.shape[1])
                    aug_image      = aug_image.reshape(fg_mask.shape[0],fg_mask.shape[1],3)
                    intersect_mask = np.logical_and(fg_mask == 255,  aug_mask== 255)
                    
                if (np.sum(intersect_mask) > int(2 / 3 * np.sum(aug_mask == 255))):
                    # when most part of aug_mask is in the fg_mask region 
                    # copy the augmentated anomaly area to the normal image
                    #n_image[aug_mask == 255, :] = aug_image[aug_mask == 255, :]
                    #n_image.setflags(write=1)
                    n_image[aug_mask == 255, :] = aug_image[aug_mask == 255, :]
                    if np.sum(aug_mask)!=0 and repeat<5:
                        return n_image, np.expand_dims(aug_mask, axis=-1)
                    else:
                        n_image, aug_mask = img_mask_extraction(n_image)
                        if np.sum(aug_mask)==0:
                            print("no anomaly inserted")
                        return n_image, np.expand_dims(aug_mask, axis=-1)
                else:
                    contours, _ = cv2.findContours(aug_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    center_xs, center_ys = [], []
                    widths, heights = [], []
                    for i in range(len(contours)):
                        M = cv2.moments(contours[i])
                        if M['m00'] == 0:  # error case
                            x_min, x_max = np.min(contours[i][:, :, 0]), np.max(contours[i][:, :, 0])
                            y_min, y_max = np.min(contours[i][:, :, 1]), np.max(contours[i][:, :, 1])
                            center_x = int((x_min + x_max) / 2)
                            center_y = int((y_min + y_max) / 2)
                        else:
                            center_x = int(M["m10"] / M["m00"])
                            center_y = int(M["m01"] / M["m00"])
                        center_xs.append(center_x)
                        center_ys.append(center_y)
                        x_min, x_max = np.min(contours[i][:, :, 0]), np.max(contours[i][:, :, 0])
                        y_min, y_max = np.min(contours[i][:, :, 1]), np.max(contours[i][:, :, 1])
                        width, height = x_max - x_min, y_max - y_min
                        widths.append(width)
                        heights.append(height)
                    if len(widths) == 0 or len(heights) == 0:  # no contours
                        n_image[aug_mask == 255, :] = aug_image[aug_mask == 255, :]
                        if np.sum(aug_mask)!=0 and repeat<5:
                            return n_image, np.expand_dims(aug_mask,axis=-1)
                        else:
                            n_image, aug_mask = img_mask_extraction(n_image)
                            if np.sum(aug_mask)==0:
                                print("no anomaly inserted")
                            return n_image, np.expand_dims(aug_mask,axis=-1)
                    else:
                        max_width, max_height = np.max(widths), np.max(heights)
                        center_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                        center_mask[int(max_height/2):img_height-int(max_height/2), int(max_width/2):img_width-int(max_width/2)] = 255
                        fg_mask = np.logical_and(fg_mask == 255, center_mask == 255)

                        x_coord = np.arange(0, img_width)
                        y_coord = np.arange(0, img_height)
                        xx, yy = np.meshgrid(x_coord, y_coord)
                        # coordinates of fg region points
                        xx_fg = xx[fg_mask]
                        yy_fg = yy[fg_mask]
                        xx_yy_fg = np.stack([xx_fg, yy_fg], axis=-1)  # (N, 2)
                        if np.random.randint(2)==0:
                            val_1    = np.random.randint(n_image.shape[0])
                            val_2    = np.random.randint(n_image.shape[0])
                            xx_yy_fg = np.array([[val_1, val_2]])
                        if xx_yy_fg.shape[0] == 0:  # no fg
                            n_image[aug_mask == 255, :] = aug_image[aug_mask == 255, :]
                            if np.sum(aug_mask)!=0 and repeat<5:
                                return n_image, np.expand_dims(aug_mask, axis=-1)
                            else:
                                n_image, aug_mask = img_mask_extraction(n_image)
                                if np.sum(aug_mask)==0:
                                    print("no anomaly inserted")
                                return n_image, np.expand_dims(aug_mask, axis=-1)

                        aug_mask_shifted = np.zeros((img_height, img_width), dtype=np.uint8)
                        for i in range(len(contours)):
                            aug_mask_shifted_i = np.zeros((img_height, img_width), dtype=np.uint8)
                            new_aug_mask_i = np.zeros((img_height, img_width), dtype=np.uint8)
                            # random generate a point in the fg region
                            idx = np.random.choice(np.arange(xx_yy_fg.shape[0]), 1)
                            rand_xy = xx_yy_fg[idx]
                            delta_x, delta_y = center_xs[i] - rand_xy[0, 0], center_ys[i] - rand_xy[0, 1]
                            
                            x_min, x_max = np.min(contours[i][:, :, 0]), np.max(contours[i][:, :, 0])
                            y_min, y_max = np.min(contours[i][:, :, 1]), np.max(contours[i][:, :, 1])
                            
                            # mask for one anomaly region
                            aug_mask_i = np.zeros((img_height, img_width), dtype=np.uint8)
                            aug_mask_i[y_min:y_max, x_min:x_max] = 255
                            aug_mask_i = np.logical_and(aug_mask == 255, aug_mask_i == 255)
                            
                            # coordinates of orginal mask points
                            xx_ano, yy_ano = xx[aug_mask_i], yy[aug_mask_i]
                            
                            # shift the original mask into fg region
                            xx_ano_shifted = xx_ano - delta_x
                            yy_ano_shifted = yy_ano - delta_y
                            outer_points_x = np.logical_or(xx_ano_shifted < 0, xx_ano_shifted >= img_width) 
                            outer_points_y = np.logical_or(yy_ano_shifted < 0, yy_ano_shifted >= img_height)
                            outer_points = np.logical_or(outer_points_x, outer_points_y)
                            
                            # keep points in image
                            xx_ano_shifted = xx_ano_shifted[~outer_points]
                            yy_ano_shifted = yy_ano_shifted[~outer_points]
                            aug_mask_shifted_i[yy_ano_shifted, xx_ano_shifted] = 255
                            
                            # original points should be changed
                            xx_ano = xx_ano[~outer_points]
                            yy_ano = yy_ano[~outer_points]
                            new_aug_mask_i[yy_ano, xx_ano] = 255
                            # copy the augmentated anomaly area to the normal image
                            n_image[aug_mask_shifted_i == 255, :] = aug_image[new_aug_mask_i == 255, :]
                            aug_mask_shifted[aug_mask_shifted_i == 255] = 255
                            aug_mask    =   np.copy(aug_mask_shifted)
                        if np.sum(aug_mask)!=0 and repeat<5:
                            return n_image, np.expand_dims(aug_mask,axis=-1)
                        else:
                            n_image, aug_mask = img_mask_extraction(n_image)
                            if np.sum(aug_mask)==0:
                                print("no anomaly inserted")
                            return n_image, np.expand_dims(aug_mask,axis=-1)
            else:  # no fg restriction
                # copy the augmentated anomaly area to the normal image
                n_image[aug_mask == 255, :] = aug_image[aug_mask == 255, :]
                if np.sum(aug_mask)!=0 and repeat<5:
                    return n_image, np.expand_dims(aug_mask, axis=-1)
                else:
                    n_image, aug_mask = img_mask_extraction(n_image)
                    if np.sum(aug_mask)==0:
                        print("no anomaly inserted")
                    return n_image, np.expand_dims(aug_mask, axis=-1)

    def simplex_noise_anomlay(self, image):
        '''
        Generate simplex noise patterns and paste at random positions into normal image
        
        Args:
            image (PIL.Image, mode=RGB) or numpy image: The original image.
        Returns:
            numpy.ndarray: (anomalous_image,mask): The image with the anomaly inserted and segmentation mask. 
        '''
        mask    =   np.zeros_like(np.asarray(image))
        count   =   0
        while np.sum(mask)==0 and count<=5:
            try: orig_h,orig_w = image.size
            except: orig_h,orig_w,_ = image.shape
            orig_img        =   np.copy(np.asarray(image))        
            mask            =   np.zeros_like(np.asarray(image))
            lower_value     =   10
            img             =   (np.asarray(image).copy()/255)
            h_,w_           =   Image.fromarray(np.asarray(image)).size
            try:
                if lower_value>h_ or lower_value>w_:
                    lower_value = 1
                    h_noise     = np.random.randint(lower_value, int(img.shape[0])-1)
                    w_noise     = np.random.randint(lower_value, int(img.shape[1])-1) 
                else:
                    h_noise     = np.random.randint(lower_value, int(img.shape[0]//2))
                    w_noise     = np.random.randint(lower_value, int(img.shape[1]//2))
                start_h_noise   = np.random.randint(1, img.shape[0] - h_noise)
                start_w_noise   = np.random.randint(1, img.shape[1] - w_noise)
            except:
                w_noise, h_noise                = 1, int(w_*0.8)
                start_h_noise, start_w_noise    = 1, 1
            # noise_size      = (h_noise, w_noise)
            # --- > simplex_noise = rand_3d_octaves(img, (3, *noise_size), 6, 0.6)
            rng             =   np.random.default_rng()
            ix, iy, iz      =   rng.random(w_noise), rng.random(h_noise), rng.random(3)
            simplex_noise   =   noise3array_fast(ix, iy, iz)   
            init_zero       =   np.zeros((img.shape[0],img.shape[1],3))
            init_zero[start_h_noise: start_h_noise + h_noise, start_w_noise: start_w_noise+w_noise, :] = 0.2 * simplex_noise.transpose(1,2,0)
            img_noise       = ((img + init_zero)*255).astype(np.uint8)
            mask            = np.expand_dims(((abs(init_zero)>0)*255).astype(np.uint8)[:,:,0], axis=-1)
            count+=1
        if np.sum(mask)==0: 
            print("Anomaly Generation Fails: Kindly Rerun this method")
        return img_noise,mask
    
    def random_perturbation(self, image):
        '''
        Generate Gaussian noise patterns and randomly paste them onto a normal image
        
        Args:
            image (PIL.Image, mode=RGB) or numpy image: The original image.
        Returns:
            numpy.ndarray: (anomalous_image,mask): The image with the anomaly inserted and segmentation mask. 
        '''
        generate_counter    = 0
        patch_mask          = np.zeros_like(np.asarray(image))
        
        orig_h,orig_w   = Image.fromarray(np.asarray(image)).size
        image           = Image.fromarray(cv2.resize(np.asarray(image),(256,256)))
        #while np.sum(patch_mask)==0:
        image           = np.asarray(image).copy()  
        while np.sum(patch_mask)==0 and generate_counter<=5:
            # generate noise image
            noise_image     = np.random.randint(0, 255, size=image.shape).astype(np.float32) / 255.0
            

            # get bkg mask
            bkg_msk         = estimate_background(image)

            # generate random mask
            patch_number    = np.random.randint(1, 5)
            augmented_image = image

            MAX_TRY_NUMBER = 200
            coor_min_dim1 = 0
            coor_min_dim2 = 0

            coor_max_dim1 = 0
            coor_max_dim2 = 0
            for i in range(patch_number):
                try_count = 0
                patch_mask      = np.zeros(image.shape[:2], dtype=np.float32)
                while try_count < MAX_TRY_NUMBER:
                    try_count += 1
                    try:
                        patch_dim1 = np.random.randint(image.shape[0] // 40, image.shape[0] // 10)
                        patch_dim2 = np.random.randint(image.shape[1] // 40, image.shape[1] // 10)
                    except: 
                        patch_dim1 = np.random.randint(1, 2)
                        patch_dim2 = np.random.randint(1, 2)
                        
                    center_dim1 = np.random.randint(patch_dim1, image.shape[0] - patch_dim1)
                    center_dim2 = np.random.randint(patch_dim2, image.shape[1] - patch_dim2)

                    
                    if bkg_msk[center_dim1, center_dim2] > 0:
                        continue

                    coor_min_dim1 = np.clip(center_dim1 - patch_dim1, 0, image.shape[0])
                    coor_min_dim2 = np.clip(center_dim2 - patch_dim2, 0, image.shape[1])

                    coor_max_dim1 = np.clip(center_dim1 + patch_dim1, 0, image.shape[0])
                    coor_max_dim2 = np.clip(center_dim2 + patch_dim2, 0, image.shape[1])

                    

                    patch_mask[coor_min_dim1:coor_max_dim1, coor_min_dim2:coor_max_dim2] = 1.0

                patch_mask                      = patch_mask[:, :, np.newaxis]
                augmented_image                 = ((augmented_image*(1-patch_mask)) + ((patch_mask*noise_image)*255)).astype(np.uint8)
                patch_mask                      = (patch_mask*255).astype(np.uint8)
                generate_counter+=1
                if np.sum(patch_mask)!=0:
                    break
        if np.sum(patch_mask)==0: 
            threshold = int(0.04*image.shape[0])
            initial_x = np.random.choice(np.arange(image.shape[0]-threshold),1)[0]
            final_x   = np.random.choice(np.arange(threshold,int(image.shape[0]*0.1)),1)[0]
            if final_x==0 or ((final_x+initial_x)==image.shape[0]):
                final_x = np.ceil(4*0.02)
            patch_mask[initial_x:initial_x+final_x, initial_x:initial_x+final_x,:] = 1
            augmented_image                 = ((augmented_image*(1-patch_mask)) + ((patch_mask*noise_image)*255)).astype(np.uint8)
            patch_mask                      = (patch_mask*255).astype(np.uint8)
            
            if np.sum(patch_mask)==0:
                print("Anomaly Generation Fails: Kindly Rerun this method")
        augmented_image, patch_mask         = cv2.resize(augmented_image,(orig_h,orig_w)), cv2.resize(patch_mask,(orig_h,orig_w)) 
        return augmented_image, patch_mask
    
    def hard_aug_cutpaste(self, image):
        '''
        Insert an anomaly into the image by applying different augmenations to normal image act 
        as anomaly source image to paste into normal image
        
        Args:
            image (PIL.Image, mode=RGB) or numpy image: The original image.
        Returns:
            numpy.ndarray: (anomalous_image,mask): The image with the anomaly inserted and segmentation mask. 
        '''
        try: orig_h,orig_w = image.size
        except: orig_h,orig_w,_ = image.shape
        orig_img                =   np.copy(np.asarray(image))
        msk                     =   np.zeros_like(np.asarray(image))
        count                   =   0
        while np.sum(msk)==0 and count<=5:
            img         =   np.asarray(image).copy()
            angle       =   np.random.choice([0, 90, 180, 270],1)[0]
            img_rot     =   ndimage.rotate(img, angle)
            img_perm    =   perm(img_rot)
            img_jit     =   cv2.resize(color_jitter(img_perm),(img.shape[1],img.shape[0]))
            
            try: 
                aug_img,msk =   crop_portion_insert_normal(img, img_jit)
            except:
                height,width,_ = img.shape
                aug_img,msk  = crop_portion_insert_normal(img, img_jit)
            count+=1
        if np.sum(msk)==0: 
            print("Anomaly Generation Fails: Kindly Rerun this method")
        return aug_img, msk

    def fract_aug(self, image, initial_point=(0,0), iterations=10000, anom_source_img=None):
        '''
        Insert an anomaly into the image by generating and pasting fractal patterns
        
        Args:
            image (PIL.Image, mode=RGB) or numpy array: The original image
        Returns:
            numpy.ndarray: (anomalous_image,mask): The image with the anomaly inserted and segmentation mask 
        '''
        orig_w, orig_h          = image.size
        image                   = Image.fromarray(np.asarray(image)) 
        buf                     = io.BytesIO()
        transformations         = [transformation1, transformation2, transformation3, transformation4]
        #probabilities           = [0.85, 0.07, 0.07, 0.01] 
        mask_img                    = np.zeros_like(image)
        mask                        = np.zeros_like(mask_img)
        count                   =   0
        while np.sum(mask)==0 and count<=5:
        #while (np.sum(mask_img)<=0.2*(np.max(mask_img.shape))):
            points = generate_points(initial_point, iterations, transformations)
            x_vals, y_vals          = zip(*points)
            plt.scatter(x_vals, y_vals, s=1, c='green')
            plt.axis('off')
            fig             = plt.gcf() 
            frac_img        = cv2.resize(cv2.cvtColor(np.asarray(fig2img(fig)),cv2.COLOR_BGR2GRAY),(np.asarray(image).shape[0],np.asarray(image).shape[1]))            
            mask_img        =   np.copy(frac_img<255)
            
            img             =   np.asarray(anom_source_img).copy()
            img_jitter      =   color_jitter(img)
            img_cut_pil     =   Image.fromarray(img_jitter)       
            angle           =   np.random.choice([0, 90, 180, 270],1)[0]
            img_rot         =   ndimage.rotate(img_cut_pil, angle)
            img_noise       =   random_noise(img_rot)
            frac_patch      =   cv2.resize(frac_img, (img_noise.shape[1], img_noise.shape[0]))  #[roi[0]:roi[1],roi[2]:roi[3]]>0
            anom_img        =   ((img_noise*np.expand_dims((frac_patch)<255, axis=-1))*255).astype(np.uint8)
                
            roi_region,roi_new                =   crop_norml_img_portion(np.asarray(image))
            anom_patch                        =   cv2.resize(anom_img,(roi_region.shape[1],roi_region.shape[0]))
            mask_img                          =   np.expand_dims(cv2.resize(mask_img.astype(np.uint8),(anom_patch.shape[1],anom_patch.shape[0])), axis=-1)
            mask                              =   np.zeros_like(image)
            mask[roi_new[0]:roi_new[1],roi_new[2]:roi_new[3],0:1] =  (mask_img*255).astype(np.uint8)
            image                             =   np.copy(np.asarray(image))
            image[roi_new[0]:roi_new[1], roi_new[2]:roi_new[3],:] =  ((1-mask_img)*image[roi_new[0]:roi_new[1], roi_new[2]:roi_new[3],:]) + (mask_img*anom_patch).astype(np.uint8)
            count+=1
        if np.sum(mask)==0: 
            print("Anomaly Generation Fails: Kindly Rerun this method")
        return image, mask[:,:,0:1]

    def cutpaste(self, image, area_ratio = (0.2, 0.25), aspect_ratio = ((0.3, 1) , (1, 3.3))):
        '''
        CutPaste augmentation
        Args:
            image: [PIL] - original image
            area_ratio: [tuple] - range for area ratio for patch
            aspect_ratio: [tuple] -  range for aspect ratio

        Returns:
            numpy.ndarray: (anomalous_image,mask): The image with the anomaly inserted and segmentation mask. 
        '''
        mask                =  np.zeros_like(np.asarray(image))
        count               =   0
        while np.sum(mask)==0 and count<=5:
            image_np            =  np.asarray(image)
            patch_w, patch_h    = patch_dim_extraction(image_np, area_ratio, aspect_ratio)
            
            img_h, img_w        = image.size
            if img_h<patch_h:
                patch_h          = img_h
            if img_w<patch_w:
                patch_w          = img_w
            try:    
                aug_img, mask   = crop_and_paste_(image, patch_w, patch_h, self.transform, rotation = False)
            except:
                image           = Image.fromarray(cv2.resize(np.asarray(image),(256,256)))
                patch_w, patch_h    = patch_dim_extraction(np.asarray(image), area_ratio, aspect_ratio)
                img_h, img_w        = image.size
                if img_h<patch_h:
                    patch_h          = img_h
                if img_w<patch_w:
                    patch_w          = img_w
                
                aug_img, mask   = crop_and_paste_(image, patch_w, patch_h, self.transform, rotation = False)
                aug_img, mask   = cv2.resize(aug_img,(img_w,img_h)), cv2.resize(mask,(img_w,img_h))
            count+=1
        
        if np.sum(mask)==0: 
            print("Anomaly Generation Fails: Kindly Rerun this method")
        return aug_img, mask
    
    def affined_anomlay(self,image, resize=(256,256),anom_source_img=None):
        '''
        Insert an anomaly into the image by converting into patches and adding
        affine transform on normal patches to get the anomaly image patches
        Args:
            image (PIL.Image, mode=RGB) or numpy array: The original image.
        Returns:
            numpy.ndarray: (anomalous_image,mask): The image with the anomaly inserted and segmentation mask. 
        '''
        if resize is None:
            resize  =   (image.size[0],image.size[1])    
        
        choice_of_patch_size                        =   np.random.choice([16,32,64],1)[0] 
        
        resize_req                              =   True
        orig_h, orig_w                          =   image.size
        resize                                  =   (256,256) 
        image                                   =   Image.fromarray(cv2.resize(np.asarray(image),(256,256)))
        patches_orig, normal_img_patches        =   patches_extrac_patches_to_img(image, patch_size=choice_of_patch_size)
        mask                                   =   np.zeros_like(normal_img_patches) ### mask color -- 1
        
        patch           = np.zeros((patches_orig.shape[-2],patches_orig.shape[-3],3)) 
        msk_patch       = np.zeros((patches_orig.shape[-2],patches_orig.shape[-3],3)) 
        mask_patches    = np.zeros_like(patches_orig)
        patches_orig    = np.copy(patches_orig)
        patch_indx  =   0
        for patch_indx in range(normal_img_patches.shape[0]):
            #normal_patch_roi        =   image_roi[i, :,:,:]
            msk                     =   mask[patch_indx, :,:,:].astype(np.float32)
            normal_patch            =   (normal_img_patches[patch_indx, :,:,:]) # /255) #.astype(np.float32)
            msk                     =   mask[patch_indx, :,:,:].astype(np.float32)            
            choice_for_anomly   =   np.random.choice([0,1],1)[0] # (2,2) # self.anomaly_type
            
            if choice_for_anomly==0: # and (np.var(normal_patch)>0.001): 
                augmented_image, msk    =   affine_transform_anomaly(normal_patch, normal_img_patches,choice_of_aug=False) # , anom_source_patch_c)
            else:
                augmented_image, msk    =   normal_patch, np.zeros_like(normal_patch)
        
            row_index	            =	patch_indx//patches_orig.shape[0]
            col_index	            =	patch_indx%patches_orig.shape[1]
            patch[:,:,:]	        =	normal_img_patches[patch_indx, :, :,:]
            msk_patch[:,:,:]        =   mask[patch_indx, :, :,:]

            patches_orig[row_index, col_index, 0,:,:,:]	    =	augmented_image
            mask_patches[row_index, col_index, 0,:,:,:]	    =	msk

                   
        aug_img 			        =   unpatchify(patches_orig, (resize[0],resize[1],3))   
        aug_img_mask 	            =   unpatchify(mask_patches, (resize[0],resize[1],3)) 
        if resize_req:
            aug_img                 =   cv2.resize(aug_img,(orig_h, orig_w))
            aug_img_mask            =   cv2.resize(aug_img_mask,(orig_h, orig_w))
        if anom_source_img is not None:
            aug_img                 =   ((aug_img_mask>0)*anom_source_img)+((1-(aug_img_mask>0))*aug_img)
        return aug_img, aug_img_mask[:,:,:1]*255

    def affine_anom_color_change(self,image,resize=(256,256)):
        '''
        Insert an anomaly into the image by converting into patches and adding
        affine transform on normal patches and apply color tarnsformation  
        to get the anomaly image patches
        Args:
            image (PIL.Image, mode=RGB) or numpy array: The original image.
        Returns:
            numpy.ndarray: (anomalous_image,mask): The image with the anomaly inserted and segmentation mask. 
        '''
        aug_img, aug_img_mask   =   self.affined_anomlay(image, resize=resize)
        aug_img                 =   change_color(aug_img, aug_img_mask)
        return aug_img, aug_img_mask

if __name__=="__main__":

    anom_methods        =   "cutpaste_scar,simplex_noise,random_perturb,hard_aug_cutpaste,cutpaste,affined_anomaly,affined_anomaly_with_color,perlin_noise_pattern,superpixel_anomaly,perlin_with_roi_anomaly,rand_augmented_cut_paste,fract_aug"
    anom_methods        =   anom_methods.split(',')   
    anom_inserton_time  =   {}
    normal_img_paths    =   ['test_examples/normal_image/000.JPG'] #,
    all_anom_images     =   []
    anom_sourc_files             =   'test_examples/anom_source_image'
    class_name                   =    'example'
    try: anom_source_paths            =    [os.path.join(anom_sourc_files,file) for file in  os.listdir(anom_sourc_files)]
    except Exception as e: print("Anom Source Data does not Exist")
            
    for normal_img in range(100000):
        for normal_img in normal_img_paths:
            hight            =   256
            width            =   256
            anom_extraction  =   np.zeros((12,hight,width,3))
            anom_map         =   np.zeros((12,hight,width,3))  
            try:
                anom_source_index            =    np.random.choice(len(anom_source_paths),1)[0]
                anom_source_img     =   cv2.imread(anom_source_paths[anom_source_index])
                anom_source_img     =   cv2.resize(anom_source_img, (width, width))
                anom_source_img_    =   Image.fromarray(anom_source_img)
            except: anom_source_img_    =   Image.fromarray(np.ones((256,256,3),dtype=np.uint8)*255)
            i=0
            ######################################1.  Cut-Paste scar Example   ###################################
            ######### CutPaste: Self-Supervised Learning for Anomaly Detection and Localization ################
            ########################### https://arxiv.org/pdf/2104.04015.pdf ##########################################
            try:
                normal_image    =   cv2.imread(normal_img) # f'data/mvtech/{class_name}/train/good/000.png')
                normal_image    =   cv2.resize(normal_image, (width, width))
            except Exception as e: 
                print("Normal Image data does not Exist")
                normal_image    =   np.ones((256,256,3),dtype=np.uint8)*255
            image           =    Image.fromarray(normal_image)
            anom_insertion  =    Anomaly_Insertion()
            ################################# exec time ######################################
            t1  =   time()
            aug_img,msk     =    anom_insertion.cutpaste_scar(image, rotation=(20,100)) 
            try: anom_inserton_time[anom_methods[i]].append(round(time()-t1,2))
            except: anom_inserton_time[anom_methods[i]] = [(round(time()-t1,2))]
            ##################################################################################
            anom_extraction[i,:,:,:]  =    cv2.resize(aug_img,(hight, width))
            anom_map[i,:,:,0]         =    cv2.resize(msk,(hight, width)) 
            i+=1
            #######################################2.  simplex noise Anomaly Example ########################
            ######### Revisiting Reverse Distillation for Anomaly Detection ############################### 
            ######## https://openaccess.thecvf.com/content/CVPR2023/papers/Tien_Revisiting_Reverse_Distillation_for_Anomaly_Detection_CVPR_2023_paper.pdf #####
            #image           =    Image.open("test_img/0.png")
            ############################## exec time ########################################
            t1  =   time()
            aug_img,msk     =    anom_insertion.simplex_noise_anomlay(image)
            try: anom_inserton_time[anom_methods[i]] =   round(time()-t1,2)
            except: anom_inserton_time[anom_methods[i]] = [(round(time()-t1,2))]
            #################################################################################
            anom_extraction[i,:,:,:]  =    cv2.resize(aug_img,(hight,width))
            anom_map[i,:,:,0]         =    cv2.resize(msk, (hight,width))
            i+=1
            #######################################3.  Random Perturbation Example ########################
            ######### Collaborative Discrepancy Optimization for Reliable Image Anomaly Localization ############################### 
            ############################# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10034849 ########################
            #image           =    Image.open("test_img/0.png")
            ###################################### exec time ################################
            t1  =   time()
            aug_img,msk     =    anom_insertion.random_perturbation(image)
            try: anom_inserton_time[anom_methods[i]].append(round(time()-t1,2))
            except: anom_inserton_time[anom_methods[i]] =   [(round(time()-t1,2))]
            #################################################################################        
            anom_extraction[i,:,:,:]  =    cv2.resize(aug_img, (hight,width))
            anom_map[i,:,:,0]         =    cv2.resize(msk,(hight,width))
            i+=1
            #######################################4.  hard_aug_cutpaste ########################
            ######### ANOSEG: ANOMALY SEGMENTATION NETWORK USING SELF-SUPERVISED LEARNING ############################### 
            ############################# https://openreview.net/pdf?id=35-QqyfmjfP ########################
            #image           =    Image.open("test_img/0.png")
            ############################ exec time #########################################
            t1  =   time()
            aug_img,msk     =    anom_insertion.hard_aug_cutpaste(image)
            try: anom_inserton_time[anom_methods[i]].append(round(time()-t1,2))
            except: anom_inserton_time[anom_methods[i]] =   [(round(time()-t1,2))]
            ################################################################################
            anom_extraction[i,:,:,:]  =    cv2.resize(aug_img,(hight,width))
            anom_map[i,:,:,0]         =    cv2.resize(msk,(hight,width))
            i+=1
            #######################################5.  Crop and Paste Example ########################
            ######### CutPaste: Self-Supervised Learning for Anomaly Detection and Localization ###################
            ######### PNI : Industrial Anomaly Detection using Position and Neighborhood Information ##############
            ############################# https://arxiv.org/pdf/2104.04015.pdf #######################################
            #image           =    Image.open("test_img/0.png")
            ############################ exec time #########################################
            t1  =   time()
            aug_img,msk                 =    anom_insertion.cutpaste(image)
            try: anom_inserton_time[anom_methods[i]].append(round(time()-t1,2))
            except: anom_inserton_time[anom_methods[i]] =   [(round(time()-t1,2))]
            #################################################################################
            anom_extraction[i,:,:,:]    =    cv2.resize(aug_img,(hight,width))
            anom_map[i,:,:,0]           =    cv2.resize(msk,(hight,width))
            i+=1
            ######################################6. Affined Anomalay #######################################
            ############################ exec time #########################################
            t1  =   time()
            affied_anom,mask_affine      =    anom_insertion.affined_anomlay(image, resize=(hight,width))
            try: anom_inserton_time[anom_methods[i]].append(round(time()-t1,2))
            except: anom_inserton_time[anom_methods[i]] =   [(round(time()-t1,2))]
            ##################################################################################
            anom_extraction[i,:,:,:]     =    cv2.resize(affied_anom,(hight,width))
            anom_map[i,:,:,0]            =    cv2.resize(mask_affine,(hight,width))
            i+=1
            #####################################7. Affined Anomalay with color ######################################
            ############################ exec time #########################################
            t1  =   time()
            affied_anom_col,mask_affine_col =    anom_insertion.affine_anom_color_change(image,resize=(hight,width))
            try: anom_inserton_time[anom_methods[i]].append(round(time()-t1,2))
            except: anom_inserton_time[anom_methods[i]] =   [(round(time()-t1,2))]
            ##################################################################################
            anom_extraction[i,:,:,:]        =    cv2.resize(affied_anom_col,(hight,width))
            anom_map[i,:,:,0]               =    cv2.resize(mask_affine_col,(hight,width))
            i+=1
            ##########################################Source Required Methods ###############################################
            ######################################8.  Perlin Noise Example ######################################
            ######### DRÆM – A discriminatively trained reconstruction embedding for surface anomaly detection ################
            ########################### https://arxiv.org/pdf/2108.07610.pdf ##########################################
            anom_source_img           =    anom_source_img_ #color_anom_source(anom_source_img_)
            ############################ exec time #########################################
            t1  =   time()
            aug_img,msk               =    anom_insertion.perlin_noise_pattern(image, anom_source_img=anom_source_img)
            try: anom_inserton_time[anom_methods[i]].append(round(time()-t1,2))
            except: anom_inserton_time[anom_methods[i]] =   [(round(time()-t1,2))]
            ##################################################################################
            anom_extraction[i,:,:,:]  =    cv2.resize(aug_img,(hight,width))
            anom_map[i,:,:,0]         =    cv2.resize(msk,(hight,width))
            i+=1
            #######################################9.  Superpixel Anomaly Example ###############################
            ######### Two-Stage Coarse-to-Fine Image Anomaly Segmentation and Detection Model ################# 
            ########################### https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4511136 ###########################
            #image           =    Image.open("test_img/0.png")
            anom_source_img         =    anom_source_img_ # color_anom_source(anom_source_img_)
            ############################ exec time #########################################
            t1  =   time()
            aug_img,msk                =    anom_insertion.superpixel_anomaly(image, anom_source_img=anom_source_img)
            try: anom_inserton_time[anom_methods[i]].append(round(time()-t1,2))
            except: anom_inserton_time[anom_methods[i]] =   [(round(time()-t1,2))]
            ##################################################################################
            anom_extraction[i,:,:,:]  =    cv2.resize(aug_img,(hight,width))
            anom_map[i,:,:,0]         =    cv2.resize(msk,(hight,width))
            i+=1
            #######################################10.  Perlin ROI Anomaly Example ###############################
            ######### MemSeg: A semi-supervised method for image surface defect detection using differences and commonalities ### 
            ########################### https://arxiv.org/pdf/2205.00908.pdf ##########################################
            #image           =    Image.open("test_img/0.png")
            anom_source_img             =    anom_source_img_ # color_anom_source(anom_source_img_)
            ############################ exec time #########################################
            t1  =   time()
            aug_img,msk                 =    anom_insertion.perlin_with_roi_anomaly(image, anom_source_img=anom_source_img)
            try: anom_inserton_time[anom_methods[i]].append(round(time()-t1,2))
            except: anom_inserton_time[anom_methods[i]] =   [(round(time()-t1,2))]
            ##################################################################################
            anom_extraction[i,:,:,:]    =    cv2.resize(aug_img,(hight,width))
            anom_map[i,:,:,0]           =    cv2.resize(msk,(hight,width))
            i+=1
            #######################################11.  random augmented CutPaste Anomaly Example ########################
            ######### Explicit Boundary Guided Semi-Push-Pull Contrastive Learning for Supervised Anomaly Detection ### 
            ########################### https://arxiv.org/pdf/2207.01463.pdf ##########################################
            #image           =    Image.open("test_img/0.png")
            anom_source_img         =    anom_source_img_ 
            ############################ exec time #########################################
            t1  =   time()
            aug_img,msk     =    anom_insertion.rand_augmented_cut_paste(image,  anom_source_img=anom_source_img)
            try: anom_inserton_time[anom_methods[i]].append(round(time()-t1,2))
            except: anom_inserton_time[anom_methods[i]] =   [(round(time()-t1,2))]
            ##################################################################################
            anom_extraction[i,:,:,:]  =    cv2.resize(aug_img,(hight,width))
            anom_map[i,:,:,0]         =    cv2.resize(msk,(hight,width))
            i+=1
            #######################################12.  fractal anomaly generation (FAG) Example ########################
            ######### FRACTALAD: A SIMPLE INDUSTRIAL ANOMALY DETECTION METHOD USING FRACTAL ANOMALY GENERATION AND ###
            ######### ############################BACKBONE KNOWLEDGE DISTILLATION #################################### 
            ############################# https://arxiv.org/pdf/2301.12739.pdf #######################################
            #image           =    Image.open("test_img/0.png")
            ############################ exec time #########################################
            t1  =   time()
            aug_img,msk                =    anom_insertion.fract_aug(image, anom_source_img=anom_source_img)
            try: anom_inserton_time[anom_methods[i]].append(round(time()-t1,2))
            except: anom_inserton_time[anom_methods[i]] =   [(round(time()-t1,2))]
            ##################################################################################
            anom_extraction[i,:,:,:]   =    cv2.resize(aug_img,(hight,width))
            anom_map[i,:,:,0]          =    cv2.resize(msk,(hight,width))
            
            i+=1
            two_row_anom               =    int(anom_extraction.shape[0]/4)
            concte_anom_imgs           =    0
            concte_anom_masks          =    0

            const       =  10
            ones_img    =  np.ones((const,anom_extraction.shape[1],3))*255
            ones_img[:,:,1] = 50
            ones_img[:,:,2] = 200
            
            for i in range(anom_extraction.shape[0]):
                if i==0:
                    concte_anom_imgs    =   np.vstack((anom_extraction[i,:,:,:].astype(np.uint8), ones_img))
                    concte_anom_masks   =   cv2.cvtColor(anom_map[i,:,:,0].astype(np.uint8),cv2.COLOR_GRAY2RGB)
                    concte_anom_masks   =   np.vstack((concte_anom_masks, ones_img))
                else:
                    anom_imgs           =   np.vstack((anom_extraction[i,:,:,:].astype(np.uint8), ones_img))
                    anom_masks          =   cv2.cvtColor(anom_map[i,:,:,0].astype(np.uint8),cv2.COLOR_GRAY2RGB)
                    anom_masks          =   np.vstack((anom_masks, ones_img))
                    
                    concte_anom_imgs    =   np.vstack((concte_anom_imgs,anom_imgs.astype(np.uint8)))
                    concte_anom_masks   =   np.vstack((concte_anom_masks,anom_masks.astype(np.uint8)))    


            anom_complte_img    =   np.hstack((concte_anom_masks,concte_anom_imgs))
            dir_save_viz        =   'results_all/pseudo_anom_viz'
            os.makedirs(dir_save_viz, exist_ok=True)
            
            rotated_img         =   rotate(anom_complte_img,90)
            ones_img = np.ones((anom_extraction.shape[1],const,3))*255
            ones_img[:,:,1] = 50
            ones_img[:,:,2] = 200
            image               =   np.hstack((rotate(np.asarray(image),90),ones_img)).astype(np.uint8)
            image               =   cv2.resize(image,(rotated_img.shape[0], rotated_img.shape[0]))
            complete_imgs       =   np.hstack((image,rotated_img))
            cv2.imwrite(f'{dir_save_viz}/{class_name}.png',complete_imgs)
            all_anom_images.append(complete_imgs)
        #####################################################################
    '''
    first_img = all_anom_images[0]
    for indx in range(1,len(all_anom_images)):
        first_img  = np.vstack((first_img, all_anom_images[indx]))
    cv2.imwrite(f'{dir_save_viz}/all_{class_name}.png',first_img)
    ##################################################################### 
    for anom_method in anom_methods:
        print(f"Anom_insertion_time of {anom_method} = ", np.mean(anom_inserton_time[anom_method]), " sec")
    
    print("--")
    '''
    ##################################################################### 
    