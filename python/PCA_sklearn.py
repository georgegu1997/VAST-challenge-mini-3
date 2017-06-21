from OneImage import *
from mask import *
from image_analysis import construct_12_subplots

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pprint import pprint

def train_pca(pca, image_set = None):
    for i in range(len(OneImage.all_images)):
        if image_set != None and i not in image_set:
            continue
        image = OneImage.all_images[i]
        image_arr = np.array(image.image)
        line_mask = get_black_line_mask(image)
        mask = np.zeros(shape=(651,651,6))
        for i in range(6):
            mask[:,:,i] = line_mask
        masked_image = ma.masked_array(image_arr, mask)
        if locals().has_key("pixels"):
            pixels = np.concatenate((pixels, masked_image.ravel().compressed().reshape((-1,6))))
        else:
            pixels = masked_image.ravel().compressed().reshape((-1,6))
    print pixels.shape
    pca.fit(pixels)
    print(pca.explained_variance_ratio_)

def plot_12_pca(pca):
    ps = construct_12_subplots()
    for i in range(12):
        image = OneImage.all_images[i]
        image_arr = image.image.reshape((-1,6))
        result = pca.transform(image_arr).reshape(651,651,3)
        max_v = np.max(result)
        min_v = np.min(result)
        result_image = ((result - min_v) / (max_v - min_v) * 255).astype(np.uint8)
        ps[i].imshow(result_image)

def show_pca_image(index, pca):
    image = OneImage.all_images[index]
    image_arr = image.image.reshape((-1,6))
    result = pca.transform(image_arr).reshape(651,651,3)
    max_v = np.max(result)
    min_v = np.min(result)
    result_image = ((result - min_v) / (max_v - min_v) * 255).astype(np.uint8)
    plt.imshow(result_image)

def main():
    read_all_images()
    pca = PCA(n_components=3, copy=True)
    train_pca(pca)
    plot_12_pca(pca)
    #show_pca_image(5, pca)
    plt.show()


if __name__ == "__main__":
    main()
