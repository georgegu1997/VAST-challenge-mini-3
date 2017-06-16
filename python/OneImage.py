import numpy as np

class OneImage:
    all_images = []

    def __init__(self, arr, name = "unnamed", dt = None):
        '''an numpy array: storing the image data'''
        self.image = arr
        '''str: storing the name of the image file'''
        self.name = name
        '''datetime object: storing the time of the shooting'''
        self.dt = dt

        OneImage.all_images.append(self)

    def get_band(self, band_index):
        '''band_index will start from 1'''
        return self.image[:,:,band_index - 1]

    def reconstruct_RGB(self, R_index, G_index, B_index):
        shape = self.image.shape
        new_arr = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        new_arr[:,:,0] = self.get_band(R_index)
        new_arr[:,:,1] = self.get_band(G_index)
        new_arr[:,:,2] = self.get_band(B_index)
        return new_arr

    def true_RGB(self):
        '''in the tif, band 1 is B, 2 is G, 3 is R'''
        return self.reconstruct_RGB(3,2,1)

    def get_NDVI(self):
        B4 = self.get_band(4).astype(np.float64)
        B3 = self.get_band(3).astype(np.float64)
        #print B3.dtype
        #print B3.shape
        NDVI = (B4-B3) / (B4+B3 + (B4+B3 == 0).astype(np.float64))
        return NDVI
