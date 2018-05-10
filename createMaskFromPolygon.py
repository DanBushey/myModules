'''
Created on May 2, 2017

@author: Surf32
'''
import numpy as np
from skimage.draw import polygon
import matplotlib.pyplot as plt

pathfile = r"C:\Users\Surf32\Desktop\ResearchDSKTOP\DataJ\A\A30_FastROI\SampleData\MaskNumpy.npy"
dict1 = np.load(pathfile)
dict1 = dict1.item()

roi = '0'
image = np.zeros((dict1['image_shape'][1]*dict1['image_shape'][2]*dict1['image_shape'][3],  dict1['image_shape'][0]), dtype=dict1['intensity_data']['0'][0].dtype)
image[dict1['mask_index'][roi][0]] = dict1['intensity_data'][roi]
image2 = image.reshape(dict1['image_shape'][1], dict1['image_shape'][2], dict1['image_shape'][3], dict1['image_shape'][0])

layer = 11
plt.imshow(np.max(image2[layer, :, :, :], axis = 2), cmap='gray')



plt.imshow(dict1['mask_index']['0'][2, :, :],  cmap='gray')






image = np.zeros(dict1['image_shape'][2:4], dtype = np.bool)
xy = dict1['roi_index']['1'][0]
xy = np.vstack(xy)
xx, yy = polygon(xy[:,0], xy[:,1])
image[yy,xx] = 1

plt.imshow(image, cmap='gray')
