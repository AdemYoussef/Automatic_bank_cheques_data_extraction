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

#Standarazing all images to match models training simples
def Prepare_arabic_words(images):
    ROI_arabic_words = 0
    for image in images:
        # append the image
        orignal_arabic_word = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Remove black border by cropping
        h = orignal_arabic_word.shape[0]
        w = orignal_arabic_word.shape[1]
        y = 3
        x = 3
        crop_image_word = orignal_arabic_word[y:h - y, x:w - x]
        # rescale to 28 by 28 by 1
        r_w2_word = 256.0 / crop_image_word.shape[1]
        r_h2_word = 256.0 / crop_image_word.shape[0]
        dim_word = (int(crop_image_word.shape[1] * r_w2_word), int(crop_image_word.shape[0] * r_h2_word))  # both
        # Format Image
        word_resized_image = cv2.resize(crop_image_word, dim_word, interpolation=cv2.INTER_LINEAR)
        # Save the image
        cv2.imwrite(
            r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\Arabic cheques\segmented words arabic\ROI_{}.png".format(
                ROI_arabic_words), word_resized_image)
        #
        ROI_arabic_words += 1

    return True
