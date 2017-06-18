import numpy as np
import numpy.ma as ma

class OneImage:
    all_images = []

    @staticmethod
    def contrast_stretch(image_arr):
        a = 0
        b = 255
        c = image_arr.min()
        d = image_arr.max()
        cs_image = ((image_arr.astype(np.float64)-c) * (b-a) / (d-c) + a).astype(np.uint8)
        return cs_image

    def __init__(self, arr, name = "unnamed", dt = None):
        '''an numpy array: storing the image data'''
        self.image = arr
        '''str: storing the name of the image file'''
        self.name = name
        '''datetime object: storing the time of the shooting'''
        self.dt = dt

        OneImage.all_images.append(self)

    def get_band(self, band_index, cs = False):
        band = self.image[:,:,band_index - 1]
        if cs:
            masked_band = ma.masked_array(band, self.get_line_mask())
            band = OneImage.contrast_stretch(masked_band)
        '''band_index will start from 1'''
        return band

    def reconstruct_RGB(self, R_index, G_index, B_index, cs = False):
        shape = self.image.shape
        new_arr = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        new_arr[:,:,0] = self.get_band(R_index, cs)
        new_arr[:,:,1] = self.get_band(G_index, cs)
        new_arr[:,:,2] = self.get_band(B_index, cs)
        return new_arr

    def true_RGB(self, cs = False):
        '''in the tif, band 1 is B, 2 is G, 3 is R'''
        return self.reconstruct_RGB(3,2,1, cs)

    def get_NDVI(self, cs = False):
        B4 = self.get_band(4, cs = cs).astype(np.float64)
        B3 = self.get_band(3, cs = cs).astype(np.float64)
        #print B3.dtype
        #print B3.shape
        NDVI = (B4-B3) / (B4+B3 + (B4+B3 == 0).astype(np.float64))
        return NDVI

    def get_line_mask(self):
        line_mask = np.any([self.get_band(1) == 0 \
                    , self.get_band(2) == 0 \
                    , self.get_band(3) == 0], axis=0)
        ''' return type is M*N array of boolean.
            True means the pixel is on the black line, should be masked'''
        return line_mask
