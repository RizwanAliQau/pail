from pai import pai
import numpy as np
import os
import cv2

normal_img_path         =  'example_img/example_0.png' #,
anom_sourc_path         =  'anom_source_imgs'
try: anom_source_files  =  [os.path.join(anom_sourc_path,file) for file in  os.listdir(anom_sourc_path)]
except Exception as e: print("Anom Source Data does not Exist")
##############################################################################
height,width            =  256,256
try:
    anom_source_index   =  np.random.choice(len(anom_source_files),1)[0]
    anom_source_img     =  cv2.imread(anom_source_files[anom_source_index])
    anom_source_img     =  cv2.resize(anom_source_img, (width, width))
except:
    print("Anomaly Source Image data does not Exist so initialize with zero image") 
    anom_source_img     =  np.zeros((height,width,3),dtype=np.uint8)
try:
    normal_image        =  cv2.imread(normal_img_path) # f'data/mvtech/{class_name}/train/good/000.png')
    normal_image        =  cv2.resize(normal_image, (width, width))
except Exception as e: 
    print("Normal Image data does not Exist so initialize with ones image")
    normal_image        =  np.ones((height,width,3),dtype=np.uint8)*255

anom_insertion  =    pai.Anomaly_Insertion()
save_path       =    'results_all'
os.makedirs(save_path,exist_ok=True)
os.makedirs(f"{save_path}/SFI",exist_ok=True)
os.makedirs(f"{save_path}/SBI",exist_ok=True)
os.makedirs(f"{save_path}/Hybrid",exist_ok=True)

repeat_anom_gen =    1
for i in range(repeat_anom_gen):
    print(f"--- {i}/{repeat_anom_gen} ---")
    # > SFI Examples 
    # 1.CutPaste Scar
    aug_img,msk     =    anom_insertion.cutpaste_scar(normal_image) 
    msk             =    cv2.cvtColor(msk[:,:,0],cv2.COLOR_GRAY2RGB) 
    cv2.imwrite(f'{save_path}/SFI/cut_paste_scar_img_mask.png',np.hstack((aug_img,msk)))
    # 2.Simplex Noise
    aug_img,msk     =    anom_insertion.simplex_noise_anomlay(normal_image)
    msk             =    cv2.cvtColor(msk[:,:,0],cv2.COLOR_GRAY2RGB) 
    cv2.imwrite(f'{save_path}/SFI/simplex_noise_img_mask.png',np.hstack((aug_img,msk)))
    # 3.Random_perturbation
    aug_img,msk     =    anom_insertion.random_perturbation(normal_image)
    msk             =    cv2.cvtColor(msk[:,:,0],cv2.COLOR_GRAY2RGB) 
    cv2.imwrite(f'{save_path}/SFI/random_perturbation_img_mask.png',np.hstack((aug_img,msk)))
    # 4.Hard Aug Cutpaste
    aug_img,msk     =    anom_insertion.hard_aug_cutpaste(normal_image) 
    msk             =    cv2.cvtColor(msk[:,:,0],cv2.COLOR_GRAY2RGB) 
    cv2.imwrite(f'{save_path}/SFI/hard_Aug_cutpaste_img_mask.png',np.hstack((aug_img,msk)))
    # 5.Cutpaste
    aug_img,msk     =    anom_insertion.cutpaste(normal_image)
    msk             =    cv2.cvtColor(msk[:,:,0],cv2.COLOR_GRAY2RGB) 
    cv2.imwrite(f'{save_path}/SFI/Cutpaste_img_mask.png',np.hstack((aug_img,msk)))
    
    # > SBI Examples
    
    # 1.Perlin Noise Pattern
    aug_img,msk     =    anom_insertion.perlin_noise_pattern(normal_image,anom_source_img=anom_source_img) 
    msk             =    cv2.cvtColor(msk[:,:,0],cv2.COLOR_GRAY2RGB) 
    cv2.imwrite(f'{save_path}/SBI/perlin_noise_pattern_img_mask.png',np.hstack((aug_img,msk)))
    # 2.Superpixel Anomaly
    aug_img,msk     =    anom_insertion.superpixel_anomaly(normal_image,anom_source_img=anom_source_img)
    msk             =    cv2.cvtColor(msk[:,:,0],cv2.COLOR_GRAY2RGB) 
    cv2.imwrite(f'{save_path}/SBI/superpixel_anomaly_img_mask.png',np.hstack((aug_img,msk)))
    # 3.Perlin ROI Anomaly
    aug_img,msk     =    anom_insertion.perlin_with_roi_anomaly(normal_image,anom_source_img=anom_source_img)
    msk             =    cv2.cvtColor(msk[:,:,0],cv2.COLOR_GRAY2RGB) 
    cv2.imwrite(f'{save_path}/SBI/perlin_with_roi_anomaly_img_mask.png',np.hstack((aug_img,msk)))
    # 4.Random Augmented CutPaste 
    aug_img,msk     =    anom_insertion.rand_augmented_cut_paste(normal_image,anom_source_img=anom_source_img) 
    msk             =    cv2.cvtColor(msk[:,:,0],cv2.COLOR_GRAY2RGB) 
    cv2.imwrite(f'{save_path}/SBI/rand_augmented_cut_paste_img_mask.png',np.hstack((aug_img,msk)))
    # 5.Fractal Anomaly
    aug_img,msk     =    anom_insertion.fract_aug(normal_image,anom_source_img=anom_source_img)
    msk             =    cv2.cvtColor(msk[:,:,0],cv2.COLOR_GRAY2RGB) 
    cv2.imwrite(f'{save_path}/SBI/fract_aug_img_mask.png',np.hstack((aug_img,msk)))

    # > Hybrid (Proposed: SFI & SBI)
    # 1.Affined Anomaly
    AAS_or_AAC_AA = np.random.choice([0,1])
    if AAS_or_AAC_AA:
        aug_img,msk     =    anom_insertion.affined_anomlay(normal_image,anom_source_img=anom_source_img)
    else:
        aug_img,msk     =    anom_insertion.affined_anomlay(normal_image)
    msk             =    cv2.cvtColor(msk[:,:,0],cv2.COLOR_GRAY2RGB) 
    cv2.imwrite(f'{save_path}/Hybrid/affined_anomlay_aug_img_mask.png',np.hstack((aug_img,msk)))
