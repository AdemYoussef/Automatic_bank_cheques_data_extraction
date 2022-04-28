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


def get_contour_precedence_words(contour_words, cols_words):
    tolerance_factor_words = 100
    origin_words = cv2.boundingRect(contour_words)
    return ((origin_words[1] // tolerance_factor_words) * tolerance_factor_words) * cols_words + origin_words[0]


def word_cropping_kelb(cropped_image):
    words_image2 = cropped_image[145:375, 0:2300]
    # canny edge detection to prepare the image for Houghlines removal
    v_words_image2 = np.median(words_image2)
    sigma_words_image2 = 0.33
    lower_words_image2 = int(max(0, (1.0 - sigma_words_image2) * v_words_image2))
    upper_words_image2 = int(min(255, (1.0 + sigma_words_image2) * v_words_image2))
    # Add black border - detection of border touching pages
    canned_words_image2 = cv2.Canny(words_image2, upper_words_image2, lower_words_image2)
    # Detect points that form a line
    lines_img_words = cv2.HoughLinesP(canned_words_image2, 1, np.pi / 180, 150, minLineLength=10, maxLineGap=450)
    # Draw lines on the image
    for line in lines_img_words:
        x1, y1, x2, y2 = line[0]
        cv2.line(words_image2, (x1, y1), (x2, y2), (255, 0, 0), 5)
    # Erosion to increase words area for detection
    words_image2_copy = words_image2.copy()
    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((5, 6), np.uint8)

    words_image2_copy_erosion = cv2.erode(words_image2_copy, kernel, iterations=3)
    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours_words, hierarchy_words = cv2.findContours(image=words_image2_copy_erosion, mode=cv2.RETR_TREE,
                                                       method=cv2.CHAIN_APPROX_SIMPLE)
    contours_words = list(contours_words)
    contours_words.sort(key=lambda x: get_contour_precedence_words(x, words_image2_copy_erosion.shape[1]))

    # Find contours, obtain bounding box, extract and save ROI
    ROI_number_words = 0
    for c in contours_words:
        x, y, w, h = cv2.boundingRect(c)
        if 20 < h < 160 and 20 < w < 500:
            cv2.rectangle(words_image2, (x, y), (x + w, y + h), (0, 255, 0), 2)
            ROI_words = words_image2[y:y + h, x:x + w]
            cv2.imwrite(
                r'C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\kelb cheque\kelb word images\ROI_{}.png'.format(
                    ROI_number_words), ROI_words)
            ROI_number_words += 1

    return words_image2