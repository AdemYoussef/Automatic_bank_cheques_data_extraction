import numpy as np
import pandas as pd
import cv2


import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from matplotlib.pyplot import figure

import os
from utils import *


def preprocessing_before_crop_kelb(image_path):
    # Read the image path
    img_path = image_path
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    # manuall remove of extra details
    # Remove logo

    (b, g, r) = image[100, 1000]
    # trying to change the pixels values to that colore
    for i in range(0, 210):
        for j in range(0, 1740):
            image[i, j] = (b, g, r)
    # Remove OR BEARER
    (b2, g2, r2) = image[220, 2300]
    for i in range(240, 350):
        for j in range(1850, 2300):
            image[i, j] = (b2, g2, r2)
    # Attemping to remove the line under "RUPEES line" without losing data details
    (b3, g3, r3) = image[500, 750]
    for i in range(440, 510):
        for j in range(0, 1685):
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

    image_threshed = cv2.adaptiveThreshold(image_filtred1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 221,
                                           12)
    # Median Blur to remove big notaciable noise

    image_blured = cv2.medianBlur(image_threshed,
                                  5)  # 3 - 5 is the best value so far because the 7 make the 'c' of lalch (2nd word not clear)
    # cropping the image
    cropped_image = image_blured[45:514, 357:2350]
    # adding black border to start preparing for edge detaction later

    image_bordered = cv2.copyMakeBorder(cropped_image, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return image_bordered
