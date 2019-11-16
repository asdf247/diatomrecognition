# -*- coding: utf-8 -*-

#import sklearn
import numpy as np
from skimage import color
#from skimage import io
#from skimage.io import imread,imshow
#original= io.imread(r"C:\Users\Moorching\Documents\BE Project\Cocconeis.jpg")
#imshow(original)
#original.shape
#np.shape(original)[0]
#np.shape(original)[1]
#grayscaled=color.rgb2gray(original)
#imshow(grayscaled)

# importing matplotlib modules 
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
from skimage.filters import sobel
from skimage.filters import threshold_otsu
#from skimage.exposure import exposure
  
# Read Images 
img = mpimg.imread(r'C:\Users\Moorching\Documents\BE Project\Cocconeis.jpg') 
  
# Output Images 
plt.imshow(img,cmap="gray")
plt.title("Cocconeis") 

#RGB to gray
grayscaled=color.rgb2gray(img)
plt.figure()
plt.title("Grayscale_Image") 
plt.imshow(grayscaled)

#Thresholding using local method
thresh=threshold_otsu(grayscaled)
print(thresh)
binary=grayscaled>thresh
plt.figure()
plt.title("Binary Image")
plt.imshow(binary)

#Sobel filter
sobel_edge=sobel(grayscaled)
plt.figure()
plt.title("Sobel Filter")
plt.imshow(sobel_edge)

#Contrast Enhancement(can't import)
#equalize= exposure.hist(grayscaled)
#plt.figure()
#plt.title("Contrast Enhancement")

#Resizing (can't import)
#from skimage import resize
#height=edge.shape[0]/4
#width=edge.shape[1]/4
#resize=resize(edge,(height,width),anti_aliasing=True)

#Canny filter
from skimage.feature import canny
canny_edge=canny(sobel_edge,sigma=0.1)
plt.figure()
plt.title("Canny Filter")
plt.imshow(canny_edge)

#Contour
from skimage import measure
contour= measure.find_contours(binary,0.8)
plt.figure()
plt.title("Contoured_Image")
plt.imshow(contour)
