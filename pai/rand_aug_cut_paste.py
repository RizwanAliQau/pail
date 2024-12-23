import albumentations as A
import numpy as np 
import imgaug.augmenters as iaa
import cv2
def randAugmenter():
    augmentors = [
        A.ElasticTransform(alpha=250.0,always_apply=True),
        A.RandomRotate90(),
                A.Flip(),
                A.Transpose(),
                A.OpticalDistortion(p=1.0, distort_limit=1.0),
                A.OneOf([
                    A.GaussNoise(),
                ], p=0.2),
                A.OneOf([
                    A.MotionBlur(p=.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                ], p=0.2),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
                A.OneOf([
                    A.OpticalDistortion(p=0.3),
                    A.GridDistortion(p=.1),
                    A.PiecewiseAffine(p=0.3),
                ], p=0.2),
                A.OneOf([
                    A.CLAHE(clip_limit=2),
                    A.Sharpen(),
                    A.Emboss(),
                    A.RandomBrightnessContrast(),            
                ], p=0.3),
                A.HueSaturationValue(p=0.3)]

    aug_ind = np.random.choice(np.arange(1,len(augmentors)), 3, replace=False)
    aug = A.Compose([augmentors[0],
                    augmentors[aug_ind[0]],
                    augmentors[aug_ind[1]],
                    augmentors[aug_ind[2]]])
        # aug = A.Compose([self.augmentors[0], A.GridDistortion(p=1.0), self.augmentors[3], self.augmentors[1], self.augmentors[7]])
    return aug


def crop_irregular_shape(image, points):

    # Create a mask with the same dimensions as the image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask_orig = np.copy(mask)
    # Fill the mask with the polygon defined by the points
    cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
    
    # Bitwise AND operation to keep only the region defined by the mask
    cropped_image = cv2.bitwise_and(image, image, mask=mask)
    
    # Find bounding box of the polygon
    x, y, w, h = cv2.boundingRect(np.array(points, dtype=np.int32))
    
    # Crop the bounding box from the masked image
    cropped_image = cropped_image[y:y+h, x:x+w]
    mask_orig[y:y+h, x:x+w] = (cropped_image>0)[:,:,0]
    return mask_orig

def random_point(shape):
    return np.random.choice(np.arange(shape),1)[0]
def img_mask_extraction(n_image):
    aug = randAugmenter()
    shape = n_image.shape[0]
    if aug(image=n_image)['image'].all()==n_image.all():
        n_image_aug     = n_image+(np.random.random(n_image.shape)*255)
    else: n_image_aug   = aug(image=n_image)['image']
    points = [(random_point(shape), random_point(shape)), (random_point(shape), random_point(shape)), (random_point(shape), random_point(shape)), (random_point(shape), random_point(shape))]  # Define the points of the irregular shape
    aug_mask = crop_irregular_shape(n_image, points)
    n_image = ((np.expand_dims(aug_mask,axis=-1)>0)*n_image_aug)+((1-np.expand_dims(aug_mask,axis=-1)>0)*n_image)
    aug_mask    =   ((aug_mask>0)*255).astype(np.uint8)
    return n_image,aug_mask                            

