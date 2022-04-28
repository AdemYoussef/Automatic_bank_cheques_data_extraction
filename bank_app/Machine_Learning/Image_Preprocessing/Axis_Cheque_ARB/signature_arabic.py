import numpy as np
import pandas as pd
import cv2


import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from matplotlib.pyplot import figure

import os
from utils import *


def signature_extraction_AXIS_arabic(image_path):
    # --------------------------------Preprocessing the cheque image --------------------------
    # Read the image path
    img_path = image_path
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    # manuall remove of extra details
    # Remove logo
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
    scale_percent = 100  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image_colred = cv2.cvtColor(cv2.resize(image, dim), cv2.COLOR_BGR2GRAY)
    # Bilateral Filter :
    # d = 5 for diameter and  sigmaColor = sigmaSpace = 75. # d no more then 10 or we loss details in the 'S'
    image_filtred1 = cv2.bilateralFilter(image_colred, 5, 75, 75)
    # adaptiveThreshold for luminusity
    image_threshed = cv2.adaptiveThreshold(image_filtred1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 251,
                                           18)
    # Median Blur to remove big notaciable noise
    image_blured = cv2.medianBlur(image_threshed, 5)
    # -------------------------------- End Preprocessing the cheque image --------------------------
    # cropping the image
    sig_Cropped = image_blured[2500:3250, 6750:8900]
    # Erosion to increase sig area for detection
    sig_Cropped_copy = sig_Cropped.copy()
    # Taking a matrix of size 8 as the kernel
    kernel = np.ones((2, 18), np.uint8)
    sig_Cropped_copy_erosion = cv2.erode(sig_Cropped_copy, kernel, iterations=4)
    thresh = cv2.threshold(sig_Cropped_copy_erosion, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Find contours, obtain bounding box, extract and save ROI
    cnts_sig = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours organisation
    cnts_sig = cnts_sig[0] if len(cnts_sig) == 2 else cnts_sig[1]
    # contours extraction
    for c in cnts_sig:
        x, y, w, h = cv2.boundingRect(c)
        if 100 < h < 550 and 100 < w < 900:
            cv2.rectangle(sig_Cropped, (x, y), (x + w, y + h), (0, 0, 0), 2)
            sig_extracted = sig_Cropped[y:y + h, x:x + x]
            cv2.imwrite(
                r'C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\Arabic cheques\signature image AXIS arabic\sig_axis_ARB.png',
                sig_extracted)

    return sig_extracted


'''
* works with fix4 and Fix2
* doesn't work with fix1 and 3
* UnboundLocalError: local variable referenced before assignment 
* Why ??? only god knows
'''