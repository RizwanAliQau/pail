from pai import pai
import numpy as np
import os
import cv2
from PIL import Image
anom_inserton_time      =  {}
normal_img_path         =  'example_img/example_0.jpg' #,
anom_sourc_path         =  'anom_source_imgs'
class_name              =  'data'
try: anom_source_files  =  [os.path.join(anom_sourc_path,file) for file in  os.listdir(anom_sourc_path)]
except Exception as e: print("Anom Source Data does not Exist")
##############################################################################
hight,width             =  256,256
try:
    anom_source_index   =  np.random.choice(len(anom_source_files),1)[0]
    anom_source_img     =  cv2.imread(anom_source_files[anom_source_index])
    anom_source_img     =  cv2.resize(anom_source_img, (width, width))
    anom_source_img_pil =  Image.fromarray(anom_source_img)
except:
    print("Anomaly Source Image data does not Exist so initialize with ones image") 
    anom_source_img_    =  Image.fromarray(np.ones((256,256,3),dtype=np.uint8))
try:
    normal_image        =  cv2.imread(normal_img_path) # f'data/mvtech/{class_name}/train/good/000.png')
    normal_image        =  cv2.resize(normal_image, (width, width))
except Exception as e: 
    print("Normal Image data does not Exist so initialize with zeros image")
    normal_image        =  np.zeros((256,256,3),dtype=np.uint8)

anom_insertion  =    pai.Anomaly_Insertion() 