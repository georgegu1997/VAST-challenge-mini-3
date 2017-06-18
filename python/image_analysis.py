import csv
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy import misc, ndimage
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

GOOD_WHETHER = [1, 5, 6, 9, 10]

NO_CLOUD = 5

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
    return np.array(points)

def plot_special_NDVI():
    image = OneImage.all_images[1]
    image_NDVI = image.get_NDVI()
    NDVI_extreme = np.all([(image_NDVI >= -0.6) \
                    , (image_NDVI <= 0)], axis=0)

    NDVI_extreme_image = NDVI_extreme.astype(np.float64)
    mask = get_black_line_mask(image)
    NDVI_extreme_image_masked = mask_image(NDVI_extreme_image, mask)

    true_image = image.true_RGB()
    true_image_masked = mask_image(true_image, mask)

    plt.imshow(NDVI_extreme_image_masked)
    print image.name

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
        if len(points) != 0:
            ax.scatter(points[:,1], 650 - points[:,0], s=5, alpha=0.3)

def show_cloud_mask_good_whether_day(image, cs = False):
    ax = plt.axes([0.1,0.1,0.8,0.8])
    ax.set_xlim([0,650])
    ax.set_ylim([0,650])
    image_true = image.true_RGB(cs = cs)
    image_bool = get_cloud_mask_good_whether_day(image, cs = cs)
    ax.imshow(np.flipud(image_true))
    plot_points_list(bool_arr_to_list(image_bool), ax)

def get_road_mask():
    image = OneImage.all_images[5]
    B1_cs = image.get_band(1,cs =True)
    B2_cs = image.get_band(2,cs =True)
    B3_cs = image.get_band(3,cs =True)
    mask = np.all([B1_cs >130, B1_cs < 250, \
                    B2_cs >130, B2_cs < 250, \
                    B3_cs >130, B3_cs < 250,], axis = 0)
    return mask

def save_all_true_color_cs():
    for image in OneImage.all_images:
        image_true = image.true_RGB(cs = True)
        plt.imsave("./image/true_cs/"+image.name+"_true_cs.png", image_true)

def save_all_NDVI(cs = False):
    for image in OneImage.all_images:
        image_NDVI = image.get_NDVI(cs = cs)
        #print image_NDVI.shape, image_NDVI.dtype
        new_image = np.zeros((651,651,3), dtype=np.uint8)
        #print new_image.shape, new_image.dtype
        new_image[:,:,1] = ((image_NDVI > 0).astype(np.float64) * image_NDVI * 255).astype(np.uint8)
        new_image[:,:,2] = ((image_NDVI < 0).astype(np.float64) * (-1) * image_NDVI * 255).astype(np.uint8)
        if cs:
            file_name = "./image/NDVI_cs/"+image.name+"_NDVI_cs.png"
        else:
            file_name = "./image/NDVI_cs/"+image.name+"_NDVI.png"
        plt.imsave(file_name, new_image)

def save_all_432(cs = False):
    for image in OneImage.all_images:
        image_true = image.reconstruct_RGB(4,3,2, cs = cs)
        plt.imsave("./image/432_cs/"+image.name+"_432_cs.png", image_true)

def save_all_542(cs = False):
    for image in OneImage.all_images:
        image_true = image.reconstruct_RGB(5,4,2, cs = cs)
        plt.imsave("./image/542_cs/"+image.name+"_542_cs.png", image_true)

def save_all_156(cs = False):
    for image in OneImage.all_images:
        image_true = image.reconstruct_RGB(1,5,6, cs = cs)
        plt.imsave("./image/156_cs/"+image.name+"_156_cs.png", image_true)

def construct_12_subplots():
    plt.figure(figsize=(15,10))
    ps = []
    for i in range(12):
        ps.append(plt.subplot(3,4,i+1))
    return ps

def draw_12_sactter():
    ps = construct_12_subplots()
    for i in range(12):
        image = OneImage.all_images[i]
        image_NDVI = image.get_NDVI()
        image_band = image.get_band(2)
        line_mask = get_black_line_mask(image)
        lake_mask = get_lake_mask()
        total_mask = np.any([line_mask, lake_mask], axis=0)

        #ps[i].hist(image_NDVI_2.ravel().compressed(), bins = 80, alpha = 0.3, label = "Before cloud masked")
        if i in GOOD_WHETHER and i != NO_CLOUD:
            print "good whether"
            cloud_mask = get_cloud_mask_good_whether_day(image)
            total_mask = np.any([total_mask, cloud_mask], axis=0)
        road_mask = get_road_mask()
        total_mask = np.any([total_mask, road_mask], axis=0)

        image_NDVI_masked = ma.masked_array(image_NDVI, total_mask)
        image_band_masked = ma.masked_array(image_band, total_mask)

        '''#show the mask
        '''
        #ps[i].imshow(total_mask.astype(np.float64))
        '''#show the hist
        '''
        #ps[i].hist(image_NDVI_masked.ravel().compressed(), bins = 80, alpha = 0.3)
        #ps[i].set_ylim([0,30000])
        '''#show the scatter
        '''
        #image_NDVI_masked_inv = ma.masked_array(image_NDVI, np.invert(total_mask))
        #image_band_masked_inv = ma.masked_array(image_band, np.invert(total_mask))
        ps[i].scatter(image_band_masked.ravel().compressed(), image_NDVI_masked.ravel().compressed(), s=1, label = "masked")

        #ps[i].scatter(image_band_masked_inv.ravel().compressed(), image_NDVI_masked_inv.ravel().compressed(), s=1, label = "unmasked")
        '''
        ps[i].scatter(ma.masked_array(image_band, np.invert(lake_mask)).ravel().compressed(), ma.masked_array(image_NDVI, np.invert(lake_mask)).ravel().compressed(), s=1, label = "lake")
        ps[i].scatter(ma.masked_array(image_band, np.invert(road_mask)).ravel().compressed(), ma.masked_array(image_NDVI, np.invert(road_mask)).ravel().compressed(), s=1, label = "road")
        if  i in GOOD_WHETHER and i != NO_CLOUD:
            ps[i].scatter(ma.masked_array(image_band, np.invert(cloud_mask)).ravel().compressed(), ma.masked_array(image_NDVI, np.invert(cloud_mask)).ravel().compressed(), s=1, label = "cloud")
        '''
        #ps[i].scatter(ma.masked_array(image_band, np.invert(line_mask)).ravel().compressed(), ma.masked_array(image_NDVI, np.invert(line_mask)).ravel().compressed(), s=2, label = "line")
        ps[i].set_xlim([0,255])
        ps[i].set_ylim([-1,1])
        ps[i].legend()

        '''#for all data
        '''
        #NDVI_mean = image_NDVI_masked.ravel().compressed().mean()
        #ps[i].set_xlim([-1,1])
        '''#for part where NDVI > 0
        '''
        NDVI_mean = ma.masked_array(image_NDVI_masked, image_NDVI_masked <= 0).ravel().compressed().mean()
        #ps[i].set_xlim([0,1])
        print NDVI_mean
        #ps[i].axvline(NDVI_mean, color='b', linestyle='dashed', linewidth=1)

        ps[i].set_title(image.name)
        plt.suptitle("B5 vs B6 of all image with lake, line, cloud(for good whether), road masked.")

        #ps[i].scatter(image_band.ravel(), image_NDVI.ravel(), s=1)
        #ps[i].set_xlim([0,255])
        #ps[i].set_ylim([-1,1])
        print "finished drawing subplot", str(i+1)

def draw_12_ditinct():
    ps = construct_12_subplots()
    for i in range(12):
        image = OneImage.all_images[i]
        bands_distinct = np.zeros(6)
        bands_max = np.zeros(6)
        bands_min = np.zeros(6)
        bands_values = []
        for b in range(6):

            band_list = ma.masked_array(image.get_band(b+1), image.get_line_mask()).ravel().compressed()
            distinct_values = np.unique(band_list)
            bands_distinct[b] = len(distinct_values)
            bands_max[b] = band_list.max()
            bands_min[b] = band_list.min()
            for value in distinct_values:
                bands_values.append([b+1, value])

        bands_values = np.array(bands_values)
        ps[i].plot(np.arange(1, 7), bands_min, label= "min value")
        ps[i].plot(np.arange(1, 7), bands_max, label= "max value")
        ps[i].scatter(bands_values[:,0], bands_values[:,1], s=0.1)
        ps[i].plot(np.arange(1, 7), bands_distinct + bands_min, label= "distinct values")
        #ps[i].fill_between(np.arange(1, 7), bands_distinct + bands_min, bands_min, facecolor='green', interpolate=True)
        ps[i].set_ylim([-5,260])
        ps[i].set_title(image.name)
        #ps[i].legend()

    plt.suptitle("Distinct value of each band in each image.")

def main():
    '''read the image data'''
    read_all_images()

    '''
    image = OneImage.all_images[5]
    image_NDVI = image.get_NDVI()
    image_B5 = image.get_band(5)
    line_mask = get_black_line_mask(image)
    image_NDVI = ma.masked_array(image_NDVI, line_mask)
    image_B5 = ma.masked_array(image_B5, line_mask)
    lake_mask = get_lake_mask()
    image_NDVI = ma.masked_array(image_NDVI, lake_mask)
    image_B5 = ma.masked_array(image_B5, lake_mask)
    plt.scatter(image_B5.ravel(),image_NDVI.ravel(), s=1)
    '''

    draw_12_sactter()

    '''test the road mask'''
    #road_mask = get_road_mask()
    #plt.imshow(road_mask.astype(np.float64))


    #plt.imshow(np.diff(np.diff(OneImage.all_images[5].get_NDVI(), axis=-1), axis=-1), cmap='gray')


    #plot_NDVI_hist(0)
    #image_bool = get_mask()
    #plt.imshow(image_bool.astype(np.float64))
    #show_cloud_mask_good_whether_day(OneImage.all_images[8])
    plt.show()

if __name__ == "__main__":
    main()
