import numpy as np
import pandas as pd
import cv2


import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from matplotlib.pyplot import figure

import os
from utils import *

from .Pre_processing_kelb import preprocessing_before_crop_kelb

def get_contour_precedence_amount(contour_amount, cols_amount):
    tolerance_factor_amount = 50
    origin_amount = cv2.boundingRect(contour_amount)
    return ((origin_amount[1] // tolerance_factor_amount) * tolerance_factor_amount) * cols_amount + origin_amount[0]


def amount_cropping_kelb(cropped_image):
    amount_image2 = cropped_image[300:490, 1280:2085]
    # canny edge detection for amount
    v_amount_image2 = np.median(amount_image2)
    sigma_amount_image2 = 0.33
    lower_amount_image2 = int(max(0, (1.0 - sigma_amount_image2) * v_amount_image2))
    upper_amount_image2 = int(min(255, (1.0 + sigma_amount_image2) * v_amount_image2))
    # Add black border - detection of border touching the mount number
    canned_amount_image2 = cv2.Canny(amount_image2, upper_amount_image2, lower_amount_image2)
    # Detect points that form a line
    lines_amount_image2 = cv2.HoughLinesP(canned_amount_image2, 1, np.pi / 180, 300, minLineLength=300,
                                          maxLineGap=600)  # it was 800
    for line in lines_amount_image2:
        x1, y1, x2, y2 = line[0]
        cv2.line(amount_image2, (x1, y1), (x2, y2), (255, 0, 0), 9)
    # Preprocessing for the extraction of the handwritten numbers one by one
    amount_image2_copy = amount_image2.copy()
    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((4, 1), np.uint8)

    amount_image2_copy_erosion = cv2.erode(amount_image2_copy, kernel, iterations=2)
    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours_amount, hierarchy_amount = cv2.findContours(image=amount_image2_copy_erosion, mode=cv2.RETR_TREE,
                                                         method=cv2.CHAIN_APPROX_SIMPLE)
    contours_amount = list(contours_amount)
    contours_amount.sort(key=lambda x: get_contour_precedence_amount(x, amount_image2_copy_erosion.shape[1]))
    # Find contours, obtain bounding box, extract and save ROI
    ROI_number_amount = 0
    for c in contours_amount:
        x, y, w, h = cv2.boundingRect(c)
        if 20 < h < 60 and 20 < w < 50:
            cv2.rectangle(amount_image2, (x, y), (x + w, y + h), (0, 0, 0), 2)
            ROI_amount = amount_image2[y:y + h, x:x + w]
            cv2.imwrite(
                r'C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\kelb cheque\kelb amount images\ROI_{}.png'.format(
                    ROI_number_amount), ROI_amount)
            ROI_number_amount += 1

    return amount_image2