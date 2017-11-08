# Demo of the concept of using peripheral vision preprocessing
# before any convolution to identify object is large images
import sys
from os.path import expanduser
sys.path.insert(0, expanduser("~") + '/python/kerasTools')

import makemodel

from keras.models import Model
from keras.layers import Conv2D

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
import scipy
from PIL import Image

class FocusModel(makemodel.MakeModel):
    def __init__(self, input_shape=(84, 84, 3), output_shape=(14, 14, 32)):
        super().__init__(input_shape, output_shape=output_shape)
        
    def create_model(self, model_type=None, load_weights=None):
        super().create_model()

class PeripheralModel(makemodel.MakeModel):
    def __init__(self, input_shape=(84, 84, 3), output_shape=(), focus_radius=64):
        super().__init__()
        self.focus_radius = focus_radius

    def peripheral_min_split(self, x, split_size, ring_radius, focus_point):
        rows = split_size
        cols = int(max(x.shape[0], x.shape[1]) / ring_radius - self.focus_radius/2)
        slice_angle = (2*math.pi)/split_size
        peripheral = np.full((rows, cols, x.shape[2]), 63, dtype=int)

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                pixel_angle = math.atan2(j-focus_point[1], i-focus_point[0])
                output_i = int(pixel_angle // slice_angle)

                pixel_dist = math.sqrt(math.pow(i-focus_point[0], 2) + math.pow(j-focus_point[1], 2))
                output_j = int(pixel_dist / ring_radius - self.focus_radius/2)
                if output_j < 0:
                    continue
                
                for k in range(x.shape[2]):
                    if peripheral[output_i, output_j, k] > x[i, j, k]:
                        # print(x[i,j,k])
                        peripheral[output_i, output_j, k] = x[i, j, k]

        return peripheral

    def peripheral_max_split(self, x, split_size, ring_radius, focus_point):
        rows = split_size
        cols = int(math.sqrt(x.shape[0]*x.shape[0] + x.shape[1]*x.shape[1]) / ring_radius - self.focus_radius/2 +1)
        slice_angle = (2*math.pi)/split_size
        peripheral = np.zeros((rows, cols, x.shape[2]))
        mask = np.zeros((rows, cols))

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                pixel_angle = math.atan2(j-focus_point[1], i-focus_point[0])
                output_i = int(pixel_angle // slice_angle)

                pixel_dist = math.sqrt(math.pow(i-focus_point[0], 2) + math.pow(j-focus_point[1], 2))
                output_j = int(pixel_dist / ring_radius - self.focus_radius/2)
                if output_j < 0:
                    continue
                
                for k in range(x.shape[2]):
                    if peripheral[output_i, output_j, k] < x[i, j, k]:
                        peripheral[output_i, output_j, k] = x[i, j, k]
                        mask[output_i, output_j] = 1
        
        for _slice in range(rows):
            countPixels = np.count_nonzero(mask[_slice, :])
            peripheral[_slice, :, :] = scipy.misc.imresize(peripheral[_slice, :countPixels, :], (cols, x.shape[2]))

        return peripheral

    def peripheral_average_split(self, x, split_size, ring_radius, focus_point):
        rows = split_size
        cols = int(max(x.shape[0], x.shape[1]) / ring_radius - self.focus_radius/2)
        slice_angle = (2*math.pi)/split_size
        peripheral = np.zeros((rows, cols, x.shape[2]))
        average_count = np.zeros((rows, cols, x.shape[2]))

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                pixel_angle = math.atan2(j-focus_point[1], i-focus_point[0])
                output_i = int(pixel_angle // slice_angle)

                pixel_dist = math.sqrt(math.pow(i-focus_point[0], 2) + math.pow(j-focus_point[1], 2))
                output_j = int(pixel_dist / ring_radius - self.focus_radius/2)
                if output_j < 0:
                    continue

                peripheral[output_i, output_j, :] += x[i, j, :]
                average_count[output_i, output_j, :] += 1
        
        peripheral[average_count > 0] = np.divide(peripheral[average_count > 0], average_count[average_count > 0])
        return peripheral

    def peripheral_preprocess(self):
        img_input = Input(shape=input_shape)
        # self.model = 
    
if __name__ == "__main__":
    model = PeripheralModel()
    test_image = mpimg.imread(expanduser("~") + "/Pictures/Wallpapers/space-wallpaper-22.jpg")
    # test_image = mpimg.imread(expanduser("~") + "/Pictures/Wallpapers/test.png")
    # test_image = mpimg.imread(expanduser("~") + "/Pictures/Wallpapers/test2.png")
    imgplot = plt.imshow(test_image)
    # plt.show()
    # print(test_image.shape)
    peripheral = model.peripheral_max_split(test_image, 360, 4, (400, 1880)) # 800, 1280
    plt.imshow(peripheral)
    plt.show()