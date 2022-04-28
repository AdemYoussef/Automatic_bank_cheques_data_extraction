import cv2
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from matplotlib.pyplot import figure

#Loading originale images
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

#MNIST-like pre-processing

def Prepare_Date_Digits(images):
    ROI_date_digit = 0
    for image in images:
        # append the image
        orignal_digit = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Remove black border by cropping
        h = orignal_digit.shape[0]
        w = orignal_digit.shape[1]
        y = 2
        x = 2
        crop_digit = orignal_digit[y:h - y, x:w - x]
        # Grayscaling
        gray_digit = cv2.cvtColor(crop_digit, cv2.COLOR_BGR2GRAY)
        # reverse colors
        revers_digit = cv2.bitwise_not(gray_digit, mask=None)
        # ---------------------------------------------------------------------
        # Reducing pixel density for more accurate digit details and avoid having "FAT" digits
        kernel = np.ones((3, 3), np.uint8)
        eroded_amount_digit = cv2.erode(revers_digit, kernel, iterations=1)
        # ---------------------------------------------------------------------
        # rescale to 28 by 28 by 1
        r_w = 28.0 / eroded_amount_digit.shape[1]
        r_h = 28.0 / eroded_amount_digit.shape[0]
        dim = (int(eroded_amount_digit.shape[1] * r_w), int(eroded_amount_digit.shape[0] * r_h))
        # Format the new rescaled image
        # Cubic because we are dealing with round shapes
        digit_resized = cv2.resize(eroded_amount_digit, dim, interpolation=cv2.INTER_CUBIC)
        # Save the image
        cv2.imwrite(
            r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\Handwritten_Digits_Preparation\prepared_date_digits_images\ROI_{}.png".format(
                ROI_date_digit), digit_resized)
        ROI_date_digit += 1

    return True
