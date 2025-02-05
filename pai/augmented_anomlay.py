
from patchify import patchify, unpatchify
import numpy as np 
import cv2
import imgaug.augmenters as iaa
from PIL import Image, ImageDraw, ImageFilter
import random
from skimage.metrics import structural_similarity as ssim
def patches_extrac_patches_to_img(img_norm, patch_size=32):
        #no_of_patches           =   320
    #patch_size              =   int(((img_norm.shape[0]*img_norm.shape[0])/no_of_patches)**(1/2))
    #patch_size 				=	 40 #80
    patches                 = 	patchify(np.asarray(img_norm), (patch_size,patch_size,3), step=patch_size) # patch shape [2,2,3]
    img_patches_norm        = 	np.zeros((patches.shape[0]*patches.shape[1],patch_size,patch_size,3), dtype=np.float32)
    
    patch_indx=0    
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            img_patches_norm[patch_indx, :,:,:]    =   patches[i,j,0] 
            patch_indx+=1
    
    
    return patches, img_patches_norm 

def extract_anom_souce_img_patches(aug= None,anomaly_source_paths='', patch_size_=32):
    anomaly_source_idx          = torch.randint(0, len(anomaly_source_paths), (1,)).item()
    anomaly_source_path         = anomaly_source_paths[anomaly_source_idx]
    anomaly_source_img          = cv2.imread(anomaly_source_path)
    anomaly_source_img          = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))

    if aug is not None: anomaly_img_augmented       = (aug(image=anomaly_source_img)/255).astype(np.float32)
    else: anomaly_img_augmented     =  (anomaly_source_img/255).astype(np.float32) #(aug(image=anomaly_source_img)/255).astype(np.float32)
    _, anom_source_img_patches      =  self.patches_extraction(anomaly_img_augmented, patch_size=patch_size_)
    # np.random.shuffle(anom_source_img_patches)
    return anomaly_img_augmented, anom_source_img_patches

def img_aug_anomaly(image_patch):
    try:
        aug_list    =   [iaa.AdditiveGaussianNoise(scale=(0, 0.2*255)), 
                        iaa.AdditivePoissonNoise(lam=(10.0,15.0), per_channel=True),
                        #iaa.CoarseSaltAndPepper(0.05, size_percent=(0.01, 0.1)),
                        iaa.JpegCompression(compression=(95, 99)),
                        iaa.BlendAlphaSomeColors(iaa.Grayscale(1.0)),
                        iaa.GaussianBlur(sigma=(2.0, 3.0)),
                        iaa.MotionBlur(k=15),
                        iaa.MeanShiftBlur(),
                        iaa.AllChannelsCLAHE(),
                        iaa.Sharpen(alpha=(1.0), lightness=(0.75, 2.0)),
                        #iaa.Alpha((0.0, 1.0),iaa.Canny(alpha=1),iaa.MedianBlur(13)),
                        iaa.Fliplr(0.5),
                        iaa.Flipud(0.5),
                        iaa.Affine(rotate=(-45, 45)),
                        iaa.PerspectiveTransform(scale=(0.01, 0.15)),
                        iaa.ElasticTransformation(alpha=(0, 5.0), sigma=0.25),
                        iaa.Rot90((1, 3)),
                        iaa.WithPolarWarping(iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})),
                        iaa.imgcorruptlike.ShotNoise(severity=2),
                        iaa.imgcorruptlike.Fog(severity=2),
                        iaa.imgcorruptlike.Frost(severity=2),
                        iaa.imgcorruptlike.Snow(severity=2),
                        iaa.imgcorruptlike.ElasticTransform(severity=2),
                        iaa.pillike.FilterBlur(),
                        iaa.pillike.Affine(rotate=(-20, 20), fillcolor=(0, 256)),
                        iaa.Superpixels(p_replace=0.5, n_segments=64),
                        iaa.UniformVoronoi(250, p_replace=0.9, max_size=None)]
                        #iaa.Cutout(nb_iterations=(1, 5), size=0.2, squared=False, fill_mode="constant",cval=0),
        aug_ind         = np.random.choice(np.arange(len(aug_list)), 1)[0] #, replace=False)
        aug             = iaa.Sequential([aug_list[aug_ind]])
        aug_image       = aug(image=image_patch.astype(np.uint8))
        
    except: 
        aug_image       = image_patch
    
    return aug_image
                            
def affine_transform_anomaly(normal_patch, normal_img_patches, anom_patches=None, choice_of_aug=True):
    
    if choice_of_aug:
        normal_patch_aug   =   img_aug_anomaly(normal_patch)
    else: normal_patch_aug   =   normal_patch
        
    rows,cols,ch    = normal_patch.shape
    points_1        = np.random.choice(normal_patch.shape[0], 6)
    points_2        = np.random.choice(normal_patch.shape[0], 6)

    # pts1 = np.float32([[50,50],[200,50],[50,200]])
    # pts2 = np.float32([[10,100],[200,50],[100,250]])
    pts1 = np.float32([[points_1[0],points_1[1]],[points_1[2],points_1[3]],[points_1[4],points_1[5]]])
    pts2 = np.float32([[points_2[0],points_2[1]],[points_2[2],points_2[3]],[points_2[4],points_2[5]]])


    M                       = cv2.getAffineTransform(pts1,pts2)

    choice_of_patch         =   np.random.choice(normal_img_patches.shape[0],1)[0]
    normal_patch_shuf       =   np.copy((normal_img_patches[choice_of_patch, :,:,:]))
    # if anom_patches is not None:
    #     anom_source_patch       =   anom_patches #np.copy((anom_patches[choice_of_patch, :,:,:]))
    #     choice_of_affine_patch  =   np.random.choice([0,1,2], 1)[0]
    # else:
    #     choice_of_affine_patch  =   np.random.choice([0,1], 1)[0]
    choice_of_affine_patch  =   np.random.choice([0,1], 1)[0]
    if choice_of_affine_patch==0:
        #normal_patch_preprocess, _  =   self.img_aug_anomaly(normal_patch)  
        dst     = cv2.warpAffine(normal_patch_aug,M,(cols,rows))
    elif choice_of_affine_patch==1: dst     = cv2.warpAffine(normal_patch_shuf,M,(cols,rows))
    elif choice_of_affine_patch==2:
        distr_addded  =   np.random.choice([0,1], 1)[0]
        if distr_addded==0:
            A = anom_source_patch.shape[0] / 3.0
            w = 2.0 / anom_source_patch.shape[1]

            shift = lambda x: A * np.sin(2.0*np.pi*x * w)

            for i in range(anom_source_patch.shape[0]):
                anom_source_patch[:,i] = np.roll(anom_source_patch[:,i], int(shift(i)))
            #a_channel = np.ones(anom_source_patch.shape, dtype=np.float32)/2
            #anom_source_patch   =   a_channel*anom_source_patch 
        dst     = cv2.warpAffine(anom_source_patch,M,(cols,rows))

    msk     = np.expand_dims((dst>0).astype(np.float32)[:,:,0], axis=2)
    if True: #ssim(dst, (msk*normal_patch),channel_axis=2,data_range=255)<=0.75:
        augmented_image         =       ((msk * dst) + ((1-msk)*normal_patch)).astype(np.float32)
    
    else:
        augmented_image         =       normal_patch 
        msk                     =       np.zeros_like(msk)
        
    return augmented_image, msk

def color_anom_source(anom_source_img):
    anom_source_img         =    np.copy(np.asarray(anom_source_img))
    color_choice            =    np.random.choice([0,1,2], 1)[0]
    color_index             =    [0,1,2].index(color_choice)
    color_value             =    [0,1,2]
    color_value .pop(color_index)

    for i in color_value: 
        anom_source_img[:,:,i]         = 0  
    
    anom_source_img  = Image.fromarray(anom_source_img)
    return anom_source_img


def apply_opacity(image):
    # Open the image and convert it to RGBA
    image = Image.fromarray(image).convert("RGBA")

    # Convert image to a NumPy array
    data = np.array(image)

    # Generate a random opacity (alpha) layer
    random_opacity = np.random.randint(0, 256, (data.shape[0], data.shape[1]), dtype=np.uint8)

    # Replace the alpha channel with the random opacity layer
    data[..., 3] = random_opacity

    # Convert back to an image
    random_opacity_image = Image.fromarray(data, 'RGBA')
    # Drop the alpha channel
    rgb_image = np.asarray(random_opacity_image)[..., :3]
    alpha_channel = np.asarray(random_opacity_image)[..., 3]
    opac_img  = rgb_image*(np.expand_dims(alpha_channel,axis=-1)/255)
    return (opac_img).astype(np.uint8)


def apply_random_damage(image,resize=(256,256)):
    try: image               =   Image.fromarray(image)
    except: pass
    
    width, height       = image.size
    draw                = ImageDraw.Draw(image)
    effects = [
        "scratch", "hole", "cut", "squeeze", "fold", "flip", "rough", "crack", "bent", "liquid",
        "contamination", "missing_piece", "swap", "thread", "glue_stain", "oily_spot", "rough_texture"
    ]
    choice = random.choice(effects)

    if choice == "scratch":
        for _ in range(random.randint(1, 5)):
            x1, y1 = random.randint(0, width), random.randint(0, height)
            x2, y2 = random.randint(0, width), random.randint(0, height)
            draw.line((x1, y1, x2, y2), fill="gray", width=random.randint(1, 3))

    elif choice == "hole":
        for _ in range(random.randint(1, 3)):
            x, y = random.randint(0, width), random.randint(0, height)
            radius = random.randint(5, 20)
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill="black")

    elif choice == "cut":
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = random.randint(0, width), random.randint(0, height)
        draw.line((x1, y1, x2, y2), fill="black", width=random.randint(5, 10))

    elif choice == "squeeze":
        factor = random.uniform(0.8, 1.2)
        if random.choice([True, False]):
            image = image.resize((int(width * factor), height))
        else:
            image = image.resize((width, int(height * factor)))

    elif choice == "fold":
        x, y = random.randint(0, width), random.randint(0, height)
        fold_width = random.randint(1, 3)
        draw.line((x, 0, x, height), fill=(128, 128, 128, 128), width=fold_width)

    elif choice == "flip":
        x, y = random.randint(0, width // 2), random.randint(0, height // 2)
        box = (x, y, x + width // 4, y + height // 4)
        region = image.crop(box).transpose(Image.FLIP_LEFT_RIGHT)
        image.paste(region, box)

    elif choice == "rough":
        image = image.filter(ImageFilter.GaussianBlur(radius=random.randint(1, 3)))

    elif choice == "crack":
        for _ in range(random.randint(1, 3)):
            points = [(random.randint(0, width), random.randint(0, height)) for _ in range(5)]
            draw.line(points, fill="black", width=2)

    elif choice == "bent":
        bend_strength = random.uniform(0.2, 0.5)
        image = image.transform(
            image.size, Image.QUAD, 
            data=(0, 0, width, 0, int(width * (1 - bend_strength)), height, int(width * bend_strength), height)
        )

    elif choice == "liquid":
        for _ in range(random.randint(5, 10)):
            x, y = random.randint(0, width), random.randint(0, height)
            size = random.randint(10, 30)
            draw.ellipse((x, y, x + size, y + size), fill=(80, 80, 80, 128))

    elif choice == "contamination":
        for _ in range(random.randint(5, 15)):
            x, y = random.randint(0, width), random.randint(0, height)
            size = random.randint(5, 20)
            draw.ellipse((x, y, x + size, y + size), fill=(120, 80, 60, 100))

    elif choice == "missing_piece":
        x, y = random.randint(0, width - 40), random.randint(0, height - 40)
        rect_width, rect_height = random.randint(20, 40), random.randint(20, 40)
        draw.rectangle((x, y, x + rect_width, y + rect_height), fill="white")

    elif choice == "swap":
        # Define the same size for both regions to be swapped
        box_width, box_height = width // 4, height // 4
        x1, y1 = random.randint(0, width - box_width), random.randint(0, height - box_height)
        x2, y2 = random.randint(0, width - box_width), random.randint(0, height - box_height)
        
        box1 = (x1, y1, x1 + box_width, y1 + box_height)
        box2 = (x2, y2, x2 + box_width, y2 + box_height)
        
        # Crop regions to swap
        region1 = image.crop(box1)
        region2 = image.crop(box2)
        
        # Paste the swapped regions
        image.paste(region1, box2)
        image.paste(region2, box1)

    elif choice == "thread":
        for _ in range(random.randint(1, 5)):
            x1, y1 = random.randint(0, width), random.randint(0, height)
            x2, y2 = random.randint(0, width), random.randint(0, height)
            draw.line((x1, y1, x2, y2), fill=(50, 50, 50), width=1)

    elif choice == "glue_stain":
        for _ in range(random.randint(3, 7)):
            x, y = random.randint(0, width), random.randint(0, height)
            size = random.randint(10, 40)
            draw.ellipse((x, y, x + size, y + size), fill=(180, 180, 140, 120))

    elif choice == "oily_spot":
        for _ in range(random.randint(3, 8)):
            x, y = random.randint(0, width), random.randint(0, height)
            size = random.randint(20, 50)
            draw.ellipse((x, y, x + size, y + size), fill=(100, 100, 100, 70))

    elif choice == "rough_texture":
        image = image.filter(ImageFilter.GaussianBlur(radius=random.randint(3, 5)))

    deformed_img_np     =   cv2.resize(cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR),resize)

    return image,deformed_img_np