import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from matplotlib.pyplot import figure
import os
from utils import *

from .Pre_processing_Axis import preprocessing_before_crop_Axis

def get_contour_precedence_words(contour_words, cols_words):
    tolerance_factor_words = 100
    origin_words = cv2.boundingRect(contour_words)
    return ((origin_words[1] // tolerance_factor_words) * tolerance_factor_words) * cols_words + origin_words[0]


def word_cropping(cropped_image):
    words_image = cropped_image[50:490, 0:1490]
    # canny edge detection to prepare the image for Houghlines removal
    v_words_image = np.median(words_image)
    sigma_words_image = 0.33
    lower_words_image = int(max(0, (1.0 - sigma_words_image) * v_words_image))
    upper_words_image = int(min(255, (1.0 + sigma_words_image) * v_words_image))
    # Add black border - detection of border touching pages
    canned_words_image = cv2.Canny(words_image, upper_words_image, lower_words_image)
    # Detect points that form a line
    lines_img_words = cv2.HoughLinesP(canned_words_image, 1, np.pi / 180, 500, minLineLength=350,
                                      maxLineGap=700)  # it was 800
    # Draw lines on the image
    for line in lines_img_words:
        x1, y1, x2, y2 = line[0]
        cv2.line(words_image, (x1, y1), (x2, y2), (255, 0, 0),
                 8)  # 9 is THE BEST but we use 8 so i can keep max words details
    # Erosion to increase words area for detection
    words_image_copy = words_image.copy()
    # Taking a matrix of size 8 as the kernel
    kernel = np.ones((8, 9), np.uint8)
    # The first parameter is the original image,
    # kernel is the matrix with which image is
    # convolved and third parameter is the number
    # of iterations, which will determine how much
    # you want to erode/dilate a given image.
    words_image_copy_erosion = cv2.erode(words_image_copy, kernel, iterations=3)
    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours_words, hierarchy_words = cv2.findContours(image=words_image_copy_erosion, mode=cv2.RETR_TREE,
                                                       method=cv2.CHAIN_APPROX_SIMPLE)
    contours_words = list(contours_words)
    #sorted(contours_words,key=lambda x: get_contour_precedence_words(x, words_image_copy_erosion.shape[1]),reverse=True)
    contours_words.sort(key=lambda x: get_contour_precedence_words(x, words_image_copy_erosion.shape[1]))
    # Find contours, obtain bounding box, extract and save ROI
    ROI_number_words = 0
    for c in contours_words:
        x, y, w, h = cv2.boundingRect(c)
        if 55 < h < 160 and 40 < w < 500:
            cv2.rectangle(words_image, (x, y), (x + w, y + h), (0, 0, 0), 2)
            ROI_words = words_image[y:y + h, x:x + w]
            cv2.imwrite(
                r'C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\segmented words\ROI_{}.png'.format(
                    ROI_number_words), ROI_words)
            ROI_number_words += 1

    return words_image


