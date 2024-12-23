from skimage.segmentation import slic
import numpy as np

def change_color(aug_img, aug_img_mask, choose_color=None, color_value=None):
    dst           = aug_img*(aug_img_mask>0)
    segments_slic = slic(dst, n_segments=10, compactness=10, sigma=1, start_label=1)
    img_color     = np.zeros_like((dst))
    #labels        = np.unique(segments_slic)
    
    for lab in np.unique(segments_slic):
        segments_slic_n = np.expand_dims((segments_slic==lab), axis=-1) # *(dst>0))>0
        #msk             =  (segments_slic_n).astype(np.uint8)
        dst_1 = dst*segments_slic_n
        
        choose_color    =   np.random.choice([0,1],1)[0]
        if color_value is None:
            if choose_color==0: dst_1[:,:,0] = np.round((np.max(dst_1[:,:,0])-np.min(dst_1[:,:,0]))*np.random.sample())
            #choose_color    =   np.random.choice([0,1],1)[0]
            elif choose_color==1: dst_1[:,:,1] = np.round((np.max(dst_1[:,:,1])-np.min(dst_1[:,:,1]))*np.random.sample())
            #choose_color    =   np.random.choice([0,1],1)[0]
            elif choose_color==2: dst_1[:,:,2] = np.round((np.max(dst_1[:,:,2])-np.min(dst_1[:,:,2]))*np.random.sample())
        else:
            try:
                upper_limit     =   np.random.choice([10,50,100],1)[0]
                channel_choice  =   np.random.choice([0,1,2],1)[0]
                #color_value     =   [int(col) for col in color_value*np.random.random()]
                color_value        = np.asarray(color_value) #            =   (np.asarray(color_value)*np.random.uniform(upper_limit-0.3,upper_limit)).astype(np.uint8)
                dst_1[:,:,:]       =   color_value #[color_value[-1],color_value[-2],color_value[-3]] # [choose_channel], 0, 0]
            except Exception as e:
                print("color name is not valid")
                print("error is: ", e)

        img_color+=(dst_1*(segments_slic_n>0))
    
    img_color   =   (aug_img*(1-(aug_img_mask/255))) + (img_color*(aug_img_mask/255))
    return img_color.astype(np.uint8) # , msk
