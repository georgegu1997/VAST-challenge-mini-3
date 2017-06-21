from OneImage import *
from mask import *
from image_analysis import construct_12_subplots

from matplotlib.mlab import PCA
import matplotlib.pyplot as plt
from pprint import pprint

def PCA_image(image):
    image_arr = np.array(image.image)
    line_mask = get_black_line_mask(image)
    mask = np.zeros(shape=(651,651,6))
    for i in range(6):
        mask[:,:,i] = line_mask
    #masked_image = ma.masked_array(image_arr, mask)
    masked_image = image_arr
    #reshaped_image = masked_image.ravel().compressed().reshape((-1,6))
    reshaped_image = masked_image.ravel().reshape((-1,6))

    results = PCA(reshaped_image)
    pprint(results.Wt)
    pprint(results.fracs)
    max_v = np.max(results.Y)
    min_v = np.min(results.Y)
    result_image = (results.Y / max(abs(max_v), abs(min_v)) * 255).astype(np.uint8).reshape((651,651,6))[:,:,0:3]
    return result_image

def draw_12_PCA():
    ps = construct_12_subplots()
    for i in range(12):
        image = OneImage.all_images[i]
        result_image = PCA_image(image)
        ps[i].imshow(PCA_image(image))

def main():
    read_all_images()

    image = OneImage.all_images[NO_CLOUD]
    result_image = PCA_image(image)
    plt.imshow(result_image)

    plt.show()



if __name__ == "__main__":
    main()
