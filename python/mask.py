import numpy as np
import numpy.ma as ma

from OneImage import *

'''index start from 0'''
GOOD_WHETHER_LAKE = [5,6,9,10]

GOOD_WHETHER = [1, 5, 6, 9, 10]

NO_CLOUD = 5

''' All operation involving the first three band should be masked'''
def get_black_line_mask(image):
    '''input is an OneImage object'''
    RGB_extreme = np.any([image.get_band(1) == 0 \
                , image.get_band(2) == 0 \
                , image.get_band(3) == 0], axis=0)
    ''' return type is M*N array of boolean.
        True means the pixel is on the black line, should be masked'''
    return RGB_extreme

def save_all_line_mask():
    for image in OneImage.all_images:
        mask = get_black_line_mask(image)
        mask = (mask*255).astype(np.uint8)
        plt.imsave("./image/line_mask/"+image.name+".png", mask)

def get_lake_mask_using_NDVI():
    counter = []
    for index in GOOD_WHETHER_LAKE:
        image = OneImage.all_images[index]
        image_NDVI = image.get_NDVI()
        filtered_NDVI = (image_NDVI <= -0.4)
        if locals().has_key('lake_mask'):
            lake_mask = np.all([lake_mask, filtered_NDVI], axis=0)
        else:
            lake_mask = filtered_NDVI
        counter.append(np.count_nonzero(filtered_NDVI))
    #plt.plot(counter)
    #plt.figure()
    return lake_mask

def get_lake_mask_using_B5():
    counter = []
    for index in GOOD_WHETHER_LAKE:
        image = OneImage.all_images[index]
        image_B5 = image.get_band(5)
        filtered_B5 = (image_B5 < 40)
        if locals().has_key('lake_mask'):
            lake_mask = np.all([lake_mask, filtered_B5], axis=0)
        else:
            lake_mask = filtered_B5
        counter.append(np.count_nonzero(filtered_B5))
    #plt.plot(counter)
    #plt.figure()
    return lake_mask

def get_lake_mask():
    return np.all([get_lake_mask_using_B5() * get_lake_mask_using_NDVI()], axis=0)

''' These default parameters (80-130) only work for the condition without contrast stretching'''
def get_cloud_mask_good_whether_day(image, ub = 130, lb = 80, cs = False):
    image_refer = OneImage.all_images[NO_CLOUD]
    image_B1 = image.get_band(1, cs = cs)
    refer_B1 = image_refer.get_band(1, cs = cs)
    image_bool = np.all([(refer_B1 - image_B1) > lb,
                (refer_B1 - image_B1) < ub], axis=0)
    line_mask = get_black_line_mask(image)
    lake_mask = get_lake_mask()
    image_bool = np.all([image_bool, np.invert(line_mask), np.invert(lake_mask)], axis=0)
    return image_bool

def get_road_mask():
    image = OneImage.all_images[5]
    B1_cs = image.get_band(1,cs =True)
    B2_cs = image.get_band(2,cs =True)
    B3_cs = image.get_band(3,cs =True)
    mask = np.all([B1_cs >130, B1_cs < 250, \
                    B2_cs >130, B2_cs < 250, \
                    B3_cs >130, B3_cs < 250,], axis = 0)
    return mask

def get_mask():
    counter = []
    for index in GOOD_WHETHER:
        image = OneImage.all_images[index]
        image_NDVI = image.get_NDVI()
        filtered_NDVI = np.all([(image_NDVI <= 0.1), \
                        (image_NDVI >= -0.2)], axis=0)
        if locals().has_key('mask'):
            mask = np.all([mask, filtered_NDVI], axis=0)
        else:
            mask = filtered_NDVI
        counter.append(np.count_nonzero(filtered_NDVI))
    #plt.plot(counter)
    #plt.figure()
    return mask

def show_cloud_mask_good_whether_day(image, cs = False):
    ax = plt.axes([0.1,0.1,0.8,0.8])
    ax.set_xlim([0,650])
    ax.set_ylim([0,650])
    image_true = image.true_RGB(cs = cs)
    image_bool = get_cloud_mask_good_whether_day(image, cs = cs)
    ax.imshow(np.flipud(image_true))
    plot_points_list(bool_arr_to_list(image_bool), ax)
