import numpy as np
import pandas as pd
import cv2


import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from matplotlib.pyplot import figure

import os
from utils import *


def preprocessing_before_crop_CANARA(image_path):
    # Read the image path
    img_path = image_path
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    # manuall remove of extra details
    # Remove logo
    (b, g, r) = image[100, 1000]
    # trying to change the pixels values to that colore
    for i in range(0, 200):
        for j in range(0, 1820):
            image[i, j] = (b, g, r)
    # Remove OR BEARER
    (b2, g2, r2) = image[500, 750]
    for i in range(240, 350):
        for j in range(1880, 2365):
            image[i, j] = (b2, g2, r2)
    for i in range(0, 83):
        for j in range(1750, 2340):
            image[i, j] = (b, g, r)
    # Attemping to remove the line under "RUPEES line" without losing data details
    (b3, g3, r3) = image[500, 750]
    for i in range(460, 540):
        for j in range(0, 1685):
            image[i, j] = (b3, g3, r3)
    for i in range(350, 400):
        for j in range(100, 380):
            image[i, j] = (b3, g3, r3)
    # Yellowsh colored image
    scale_percent = 100  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image_colred = cv2.cvtColor(cv2.resize(image, dim), cv2.COLOR_BGR2GRAY)
    # Bilateral Filter :
    image_filtred1 = cv2.bilateralFilter(image_colred, 5, 75, 75)
    # adaptiveThreshold for luminusity
    image_threshed = cv2.adaptiveThreshold(image_filtred1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 231,
                                           14)
    # Median Blur to remove big notaciable noise
    image_blured = cv2.medianBlur(image_threshed,
                                  5)  # 3 - 5 is the best value so far because the 7 make the 'c' of lalch (2nd word not clear)
    # cropping the image
    cropped_image = image_blured[45:550, 200:2350]
    # adding black border to start preparing for edge detaction later
    image_bordered = cv2.copyMakeBorder(cropped_image, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return image_bordered
