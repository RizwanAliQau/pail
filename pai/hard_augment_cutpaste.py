import cv2
import torchvision.transforms as transforms
from PIL import Image
import numpy as np 


def perm(img):
    #Seperating the height and width from the image data
    try:
        (h, w) = img.shape[:2] 

        #Finding the center
        centerX, centerY = int(w//2), int(h//2) 

        #spliting the image
        topleft         = img[0:centerY, 0:centerX]
        topright        = img[0:centerY, centerX:w]
        bottomleft      = img[centerY:h, 0:centerX]
        bottomright     = img[centerY:h, centerX:w]

        im_list_2d      = [[topleft, bottomleft], [topright, bottomright]]
        img_perm        = cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])
    except:
        (orig_h, orig_w) = img.shape[:2] 
        img         =    cv2.resize(img,(256,256))
        (h, w) = img.shape[:2] 

        #Finding the center
        centerX, centerY = int(w//2), int(h//2) 

        #spliting the image
        topleft         = img[0:centerY, 0:centerX]
        topright        = img[0:centerY, centerX:w]
        bottomleft      = img[centerY:h, 0:centerX]
        bottomright     = img[centerY:h, centerX:w]

        im_list_2d      = [[topleft, bottomleft], [topright, bottomright]]
        img_perm        = cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d]) 
        img_perm        = cv2.resize(img_perm, (orig_h, orig_w)) 
    return img_perm

def interchange_points(x1_crop_point, x2_crop_point):
    temp            =   x1_crop_point.copy()
    x1_crop_point   =   x2_crop_point.copy()
    x2_crop_point   =   temp
    return x1_crop_point, x2_crop_point
def color_jitter(img):
    img_pil     =   Image.fromarray(img)
    transform   =   transforms.ColorJitter(brightness=0.5, contrast=1, saturation=0.1, hue=0.5)
    output_img  =   np.asarray(transform(img_pil))
    return output_img

def crop_portion_insert_normal(img, img_jit):
    range_val       =   [i for i in range(img.shape[0])]
    x1_crop_point   =   np.random.choice(range_val,1)[0]
    y1_crop_point   =   np.random.choice(range_val,1)[0]
    range_val       =   [i for i in range(img.shape[1])]
    x2_crop_point   =   np.random.choice(range_val,1)[0]
    y2_crop_point   =   np.random.choice(range_val,1)[0]

    if x1_crop_point>=x2_crop_point: 
        x1_crop_point, x2_crop_point = interchange_points(x1_crop_point, x2_crop_point)
        if x1_crop_point==x2_crop_point: x2_crop_point+=10
    if y1_crop_point>=y2_crop_point: 
        y1_crop_point, y2_crop_point = interchange_points(y1_crop_point, y2_crop_point)
        if y1_crop_point==y2_crop_point: y2_crop_point+=10

    img[x1_crop_point:x2_crop_point, y1_crop_point:y2_crop_point] = img_jit[x1_crop_point:x2_crop_point, y1_crop_point:y2_crop_point]
    mask            =   np.expand_dims(np.zeros_like(img)[:,:,0], axis=-1)
    mask[x1_crop_point:x2_crop_point, y1_crop_point:y2_crop_point] = 255 
    
    return img, mask