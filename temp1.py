# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 16:33:19 2019

@author: DELL
"""
#import packages
from skimage import io
from matplotlib import pyplot as plt

#function defined for simple image viewing
def show_image(image,title='Image',cmap_type='gray'):
    plt.imshow(image,cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()

#importing the image from or pc (r used to remove codec errors)
img = io.imread(r'C:\Users\DELL\.spyder-py3\image1.jpg')
show_image(img,"orignal")

#image grayscale
from skimage.color import rgb2gray
img2=rgb2gray(img)
show_image(img2,"grayscaled")
img2.shape

#image thresholding
from skimage.filters import threshold_local

block_size=35
local_thresh=threshold_local(img2,block_size,offset=100)

binary_local=img2 < local_thresh
show_image(binary_local,'local thresh')

from skimage.filters import threshold_otsu

thresh=threshold_otsu(img2)
binary_ots=img2 <  thresh
show_image(binary_ots,'otsu img')

#image filtering
from skimage.filters import sobel
edge_sobel = sobel(img2)
show_image(edge_sobel,"sobel_filter")
edge2 = sobel(binary_ots)
show_image(binary_ots,"otsu imagefiltered")

img3=io.imsave