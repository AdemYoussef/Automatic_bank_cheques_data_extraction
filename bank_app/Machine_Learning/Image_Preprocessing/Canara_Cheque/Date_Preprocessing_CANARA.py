import numpy as np
import pandas as pd
import cv2


import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from matplotlib.pyplot import figure

import os
from utils import *

from .Pre_processing_CANARA import preprocessing_before_crop_CANARA

def get_contour_precedence_date(contour_date, cols_date):
    tolerance_factor_date = 35
    origin_date = cv2.boundingRect(contour_date)
    return ((origin_date[1] // tolerance_factor_date) * tolerance_factor_date) * cols_date + origin_date[0]


def date_cropping_CANARA(cropped_image):
    date_image2 = cropped_image[0:180, 1510:2300]
    date_image2_copy = date_image2.copy()
    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((4, 4), np.uint8)
    date_image2_copy_erosion = cv2.erode(date_image2_copy, kernel, iterations=1)
    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours_date, hierarchy_date = cv2.findContours(image=date_image2_copy_erosion, mode=cv2.RETR_TREE,
                                                     method=cv2.CHAIN_APPROX_SIMPLE)
    # sorting contours
    contours_date = sorted(contours_date,
                           key=lambda x: get_contour_precedence_date(x, date_image2_copy_erosion.shape[1]))
    # Find contours, obtain bounding box, extract and save ROI
    ROI_number_date = 0
    for c in contours_date:
        x, y, w, h = cv2.boundingRect(c)
        if 40 < h < 60 and 40 < w < 60:
            cv2.rectangle(date_image2, (x, y), (x + w, y + h), (0, 0, 0), 2)
            ROI_date = date_image2[y:y + h, x:x + w]
            cv2.imwrite(
                r'C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\CANARA cheque\CANARA date images\ROI_{}.png'.format(
                    ROI_number_date), ROI_date)
            ROI_number_date += 1

    return date_image2