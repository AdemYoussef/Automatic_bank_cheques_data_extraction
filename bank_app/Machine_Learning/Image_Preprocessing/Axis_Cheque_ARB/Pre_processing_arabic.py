import numpy as np
import pandas as pd
import cv2


import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from matplotlib.pyplot import figure

import os
from utils import *


def preprocessing_before_crop_Axis_arabic(image_path):
    # Read the image path
    img_path = image_path
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    # manuall remove of extra details
    # Remove logo
    # To get the value of the pixel (x=50, y=50), we would use the following code
    # X= toul , Y= 3orth
    (b, g, r) = image[500, 4000]
    # trying to change the pixels values to that colore
    for i in range(0, 724):
        for j in range(0, 6890):
            image[i, j] = (b, g, r)
    # Remove OR BEARER
    (b2, g2, r2) = image[1000, 6000]
    for i in range(900, 1500):
        for j in range(7050, 9000):
            image[i, j] = (b2, g2, r2)
    # Attemping to remove the line under "RUPEES line" without losing data details
    (b3, g3, r3) = image[1750, 2000]
    for i in range(1800, 2050):
        for j in range(0, 6400):
            image[i, j] = (b3, g3, r3)
    # Yellowsh colored image
    # for resizing the image to 100%
    scale_percent = 100  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image_colred = cv2.cvtColor(cv2.resize(image, dim), cv2.COLOR_BGR2GRAY)
    # Bilateral Filter :
    # d = 5 for diameter and  sigmaColor = sigmaSpace = 75. # d no more then 10 or we loss details in the 'S'
    image_filtred1 = cv2.bilateralFilter(image_colred, 5, 75, 75)
    # adaptiveThreshold for luminusity
    # adaptiveThreshold(src, dst, maxValue, adaptiveMethod, thresholdType, blockSize, C)
    # we are going to play on the blockSize , C and 255 => make all pixels black to give more details to the words
    # lower blocksize means more noise
    # C : more C more details
    image_threshed = cv2.adaptiveThreshold(image_filtred1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 251,
                                           18)
    # Median Blur to remove big notaciable noise
    # Median blur replace center pixel by median of pixels under kelner
    # => removes thin details
    image_blured = cv2.medianBlur(image_threshed,
                                  5)  # 3 - 5 is the best value so far because the 7 make the 'c' of lalch (2nd word not clear)
    # cropping the image
    cropped_image = image_blured[0:2060, 700:9000]
    # adding black border to start preparing for edge detaction later
    # Add black border - detection of border touching pages for Canny edge in the next step
    # 0 is the black pixel and 5 for how much width in pixels the border should be
    image_bordered = cv2.copyMakeBorder(cropped_image, 3, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return image_bordered
