import numpy as np
import pandas as pd
import cv2


import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from matplotlib.pyplot import figure

import os
from utils import *


def signature_extraction_KELB(image_path):
    # --------------------------------Preprocessing the cheque image --------------------------
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
    image_filtred1 = cv2.bilateralFilter(image_colred, 5, 75, 75)
    # adaptiveThreshold for luminusity

    image_threshed = cv2.adaptiveThreshold(image_filtred1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 221,
                                           12)
    # Median Blur to remove big notaciable noise

    image_blured = cv2.medianBlur(image_threshed,
                                  5)  # 3 - 5 is the best value so far because the 7 make the 'c' of lalch (2nd word not clear)
    # -------------------------------- End Preprocessing the cheque image --------------------------
    # cropping the image
    sig_Cropped = image_blured[600:880, 1750:2250]
    # Erosion to increase sig area for detection
    sig_Cropped_copy = sig_Cropped.copy()
    # Taking a matrix of size 8 as the kernel
    kernel = np.ones((5, 5), np.uint8)
    sig_Cropped_copy_erosion = cv2.erode(sig_Cropped_copy, kernel, iterations=3)
    thresh = cv2.threshold(sig_Cropped_copy_erosion, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Find contours, obtain bounding box, extract and save ROI
    cnts_sig = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours organisation
    cnts_sig = cnts_sig[0] if len(cnts_sig) == 2 else cnts_sig[1]
    # contours extraction
    for c in cnts_sig:
        x, y, w, h = cv2.boundingRect(c)
        if 50 < h < 250 and 50 < w < 400:
            cv2.rectangle(sig_Cropped, (x, y), (x + w, y + h), (0, 0, 0), 2)
            sig_extracted = sig_Cropped[y:y + h, x:x + x]
            cv2.imwrite(
                r'C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\kelb cheque\signature image KELB\sig_kelb.png',
                sig_extracted)

    return sig_extracted
