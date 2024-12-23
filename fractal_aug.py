import matplotlib.pyplot as plt
import numpy as np
import random 
from PIL import Image
import io
def transformation1(p):
    x, y = p
    return 0.85 * x + 0.04 * y, -0.04 * x + 0.85 * y + 1.6

def transformation2(p):
    x, y = p
    return 0.20 * x - 0.26 * y, 0.23 * x + 0.22 * y + 1.6

def transformation3(p):
    x, y = p
    return -0.15 * x + 0.28 * y, 0.26 * x + 0.24 * y + 0.44

def transformation4(p):
    x, y = p
    return 0, 0.16 * y



def choose_transformation(probabilities):
    try: 
        if probabilities[-1]<0: probabilities.pop()
    except: 
        if probabilities[-1]<0: probabilities  =  probabilities[:-1]
    
    try: np.random.choice(len(probabilities), p=probabilities) 
    except: 
        if np.sum(probabilities)>1: 
            final_probal    =  np.sum(probabilities)-1
            max_index = np.argmax(probabilities)
            probabilities[max_index] = probabilities[max_index]-final_probal
        else:
            final_probal    =  1- np.sum(probabilities)
            max_index = np.argmax(probabilities)
            probabilities[max_index] = probabilities[max_index]+final_probal

    return np.random.choice(len(probabilities), p=probabilities)

def random_val_gener(init=0.005, final=0.85):
    probab      =    np.array([])
    for i in range(3):
        value  =    random.uniform(init, final)
        probab =    np.append(probab, value)    
        final  =    1- np.sum(probab)
    
    last_prob  =    1 - np.sum(probab)
    probab     =    np.append(probab, last_prob)  
    assert np.sum(probab) == 1 
    return probab

def generate_points(p, iterations, transformations_, probabilities=None):
    transformations     =   []
    if probabilities is None:
        probabilities   =   random_val_gener()
    for i in range(4):
        trans_index     =   np.random.choice([i for i in range(4)],1)[0]
        transformations.append(transformations_[trans_index])
    points = [p]
    for _ in range(iterations):
        transformation = transformations[choose_transformation(probabilities)]
        p = transformation(p)
        points.append(p)
    return points

def abs_val_norm(x_vals):
    # x_vals    = np.array(x_vals)
    # x_vals    = x_vals*(x_vals>0)
    x_abs_val = np.abs(list(x_vals))
    x_abs_val_norm = x_abs_val/np.max(x_abs_val)
    return x_abs_val_norm

def interchange_points(x1_crop_point, x2_crop_point):
    temp            =   x1_crop_point.copy()
    x1_crop_point   =   x2_crop_point.copy()
    x2_crop_point   =   temp
    return x1_crop_point, x2_crop_point

def crop_norml_img_portion(img):
    img_w, img_h        =   img.shape[0], img.shape[1]
    choice_the_upper    =   np.random.choice([0.2,0.3,0.4,0.5,0.6],1)[0]
    upper_limit         =   int(choice_the_upper*np.min([img_w,img_h]))
    
    start_point_x   =   np.random.choice(np.arange(0,img.shape[0]-upper_limit),1)[0]
    end_point_x     =   np.random.choice(np.arange(start_point_x+int(0.5*upper_limit),start_point_x+upper_limit),1)[0]
    
    start_point_y   =   np.random.choice(np.arange(0,img.shape[1]-upper_limit),1)[0]
    end_point_y     =   np.random.choice(np.arange(start_point_y+int(0.5*upper_limit),start_point_y+upper_limit),1)[0] 
    
    #range_val       =   [i for i in range(int((img.shape[0]*0.1)),img.shape[0])]
    x1_crop_point   =   start_point_x   #np.random.choice(range_val,1)[0]
    y1_crop_point   =   start_point_y      #np.random.choice(range_val,1)[0]
    x2_crop_point   =   end_point_x                #np.random.choice(range_val,1)[0]
    y2_crop_point   =   end_point_y                #np.random.choice(range_val,1)[0]

    if x1_crop_point>=x2_crop_point: 
        x1_crop_point, x2_crop_point = interchange_points(x1_crop_point, x2_crop_point)
        if x1_crop_point==x2_crop_point: x2_crop_point+=10
    if y1_crop_point>=y2_crop_point: 
        y1_crop_point, y2_crop_point = interchange_points(y1_crop_point, y2_crop_point)
        if y1_crop_point==y2_crop_point: y2_crop_point+=10
    roi     =   [x1_crop_point,x2_crop_point, y1_crop_point,y2_crop_point]
    return img[x1_crop_point:x2_crop_point, y1_crop_point:y2_crop_point], roi 




# iterations = 10000
# initial_point = (0, 0)
# points = generate_points(initial_point, iterations)

# x_vals, y_vals = zip(*points)
# plt.scatter(x_vals, y_vals, s=1, c='green')
# plt.title('Barnsley Fern Fractal')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()

def rotate_img(img):
    flip_method     =   [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]
    flip_choice     =   np.random.choice([0,1],1)[0]
    flip_img        =   img.transpose(method=flip_img[flip_choice])
    return np.asrray(flip_img)

def fig2img(fig): 
    buf = io.BytesIO() 
    fig.savefig(buf) 
    buf.seek(0) 
    img = Image.open(buf) 
    return np.asarray(img) 