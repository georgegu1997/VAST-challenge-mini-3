import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy import misc
from skimage import io
from PIL import Image

from OneImage import *

IMAGE_FILE_LIST = [
    'image01_2014_03_17',
    'image02_2014_08_24',
    'image03_2014_11_28',
    'image04_2014_12_30',
    'image05_2015_02_15',
    'image06_2015_06_24',
    'image07_2015_09_12',
    'image08_2015_11_15',
    'image09_2016_03_06',
    'image10_2016_06_26',
    'image11_2016_09_06',
    'image12_2016_12_19'
]

'''index start from 0'''
GOOD_WHETHER_LAKE = [5,6,9,10]

def read_all_images():
    for file_name in IMAGE_FILE_LIST:
        #print "reading:", file_name
        image_arr = io.imread("./image/"+file_name+".tif")
        #print image_arr.shape
        #print image_arr.dtype
        one_image = OneImage(arr=image_arr, name=file_name)
        #print image_arr[0,0]
    print "reading completed"

def plot_NDVI_hist(image_index, log = False):
    NDVI = OneImage.all_images[image_index].get_NDVI()
    print NDVI[0,0]
    NDVI_list = np.ravel(NDVI)
    print NDVI_list.shape
    plt.hist(NDVI_list, bins = 200)
    plt.title("NDVI histogram for "+IMAGE_FILE_LIST[image_index])
    plt.xlabel("NDVI")
    plt.ylabel("pixel count")
    if log:
        plt.yscale("log")

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
    return get_lake_mask_using_B5() * get_lake_mask_using_NDVI()

''' mask: True means pixel should be mask'''
def mask_image(image_arr, mask):
    shape = image_arr.shape
    mask = (np.invert(mask)).astype(np.float64)
    reshaped_mask = np.zeros(shape)
    if len(shape) > 2 and shape[2] > 1:
        reshaped_mask = np.zeros(shape)
        for i in range(shape[2]):
            reshaped_mask[:,:,i] = mask
    else:
        reshaped_mask = mask
    masked = image_arr * reshaped_mask
    #print masked.shape
    return masked

def bool_arr_to_list(arr):
    assert(arr.dtype == bool)
    shape = arr.shape
    points = []
    for x in range(shape[0]):
        for y in range(shape[1]):
            if arr[x,y]:
                points.append([x, y])
    return points

def plot_special_NDVI():
    image = OneImage.all_images[1]
    image_NDVI = image.get_NDVI()
    NDVI_extreme = np.all([(image_NDVI <= -0.6) \
                    , (image_NDVI <= 0)], axis=0)

    NDVI_extreme_image = NDVI_extreme.astype(np.float64)
    mask = get_black_line_mask(image)
    NDVI_extreme_image_masked = mask_image(NDVI_extreme_image, mask)

    true_image = image.true_RGB()
    true_image_masked = mask_image(true_image, mask)

    plt.imshow(NDVI_extreme_image_masked)
    print image.name

def plot_points_list(points, ax, rect = False):
    if rect:
        for point in points:
            rect = patches.Rectangle(
                (point[1], 650 - point[0]),
                width=1,
                height=1,
                color="lightblue",
                alpha=0.3)
            ax.add_patch(rect)
    else:
        ax.scatter(points[:,1], 650 - points[:,0], s=5, alpha=0.3)

def main():
    '''read the image data'''
    read_all_images()

    lake_mask = get_lake_mask().astype(bool)
    points = np.array(bool_arr_to_list(lake_mask))

    ax = plt.subplot(111, aspect='equal')
    ax.set_xlim([0,650])
    ax.set_ylim([0,650])
    plot_points_list(points, ax)

    '''
    lake_mask = get_lake_mask()
    plt.imshow(lake_mask)

    plt.imshow(OneImage.all_images[5].get_band(5))
    '''
    #plot_special_NDVI()
    #plot_NDVI_hist(0)
    #image = OneImage.all_images[0].true_RGB()
    #plt.imshow(image)
    plt.show()

if __name__ == "__main__":
    main()
