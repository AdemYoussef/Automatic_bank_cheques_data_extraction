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
    images_amount = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images_amount.append(img)
    return images_amount

#MNIST-like pre-processing
def Prepare_amount_Digits(images_amount):
    ROI_amount_digit = 0
    for image in images_amount:
        # append the image
        orignal_digit_amount = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Remove black border by cropping
        h = orignal_digit_amount.shape[0]
        w = orignal_digit_amount.shape[1]
        y = 3
        x = 3
        crop_digit_amount = orignal_digit_amount[y:h - y, x:w - x]
        # Grayscaling
        gray_digit_amount = cv2.cvtColor(crop_digit_amount, cv2.COLOR_BGR2GRAY)
        # reverse colors
        revers_digit_amount = cv2.bitwise_not(gray_digit_amount, mask=None)
        # ---------------------------------------------------------------------
        # Reducing pixel density for more accurate digit details and avoid having "FAT" digits
        kernel = np.ones((3, 3), np.uint8)
        eroded_amount_digit = cv2.erode(revers_digit_amount, kernel, iterations=1)
        # ---------------------------------------------------------------------
        # rescale to 28 by 28 by 1
        r_w_amount = 28.0 / eroded_amount_digit.shape[1]
        r_h_amount = 28.0 / eroded_amount_digit.shape[0]
        dim_amount = (int(eroded_amount_digit.shape[1] * r_w_amount), int(eroded_amount_digit.shape[0] * r_h_amount))
        # Format the new rescaled image
        # Cubic because we are dealing with round shapes
        digit_resized_amount = cv2.resize(eroded_amount_digit, dim_amount, interpolation=cv2.INTER_CUBIC)
        # Save the image
        cv2.imwrite(
            r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\Handwritten_Digits_Preparation\prepared_amount_digits_images\ROI_{}.png".format(
                ROI_amount_digit), digit_resized_amount)

        ROI_amount_digit += 1

    return True
