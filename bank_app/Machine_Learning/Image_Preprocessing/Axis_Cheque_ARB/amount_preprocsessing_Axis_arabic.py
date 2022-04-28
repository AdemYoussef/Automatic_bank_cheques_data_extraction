import numpy as np
import pandas as pd
import cv2


import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from matplotlib.pyplot import figure

import os
from utils import *



from .Pre_processing_arabic import preprocessing_before_crop_Axis_arabic


def get_contour_precedence_amount(contour_amount, cols_amount):
    tolerance_factor_amount = 100
    origin_amount = cv2.boundingRect(contour_amount)
    return ((origin_amount[1] // tolerance_factor_amount) * tolerance_factor_amount) * cols_amount + origin_amount[0]


def amount_cropping_arabic(cropped_image):
    amount_image = cropped_image[1600:2060, 5800:8100]
    # canny edge detection for amount
    v_amount_image = np.median(amount_image)
    sigma_amount_image = 0.33
    lower_amount_image = int(max(0, (1.0 - sigma_amount_image) * v_amount_image))
    upper_amount_image = int(min(255, (1.0 + sigma_amount_image) * v_amount_image))
    # Add black border - detection of border touching the mount number
    canned_amount_image = cv2.Canny(amount_image, upper_amount_image, lower_amount_image)
    '''
    # Detect points that form a line
    lines_amount_image = cv2.HoughLinesP(canned_amount_image,1,np.pi/180,360,minLineLength=400, maxLineGap=1000) 
    if(lines_amount_image.all() != None):   #lines_amount_image.all() because lines_amount_image is an array and we cant compare an array to a None type
        for line in lines_amount_image:
            x1, y1, x2, y2 = line[0]
            cv2.line(amount_image, (x1, y1), (x2, y2), (255, 0, 0), 14)
    else:'''
    # Preprocessing for the extraction of the handwritten numbers one by one
    amount_image_copy = amount_image.copy()
    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((5, 4), np.uint8)
    amount_image_copy_erosion = cv2.erode(amount_image_copy, kernel, iterations=3)
    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours_amount, hierarchy_amount = cv2.findContours(image=amount_image_copy_erosion, mode=cv2.RETR_TREE,
                                                         method=cv2.CHAIN_APPROX_SIMPLE)
    contours_amount = list(contours_amount)
    contours_amount.sort(key=lambda x: get_contour_precedence_amount(x, amount_image_copy_erosion.shape[1]))
    # Find contours, obtain bounding box, extract and save ROI
    ROI_number_amount = 0
    for c in contours_amount:
        x, y, w, h = cv2.boundingRect(c)
        if 100 < h < 250 and 105 < w < 250:
            cv2.rectangle(amount_image, (x, y), (x + w, y + h), (0, 0, 0), 1)
            ROI_amount = amount_image[y:y + h, x:x + w]
            cv2.imwrite(
                r'C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\Arabic cheques\segmented amount arabic\ROI_{}.png'.format(
                    ROI_number_amount), ROI_amount)
            ROI_number_amount += 1

    return amount_image