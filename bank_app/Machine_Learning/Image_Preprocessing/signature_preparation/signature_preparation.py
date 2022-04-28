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

#Standarazing all images to match the signature dataset
def Prepare_signature(images, path_to_save):
    # ROI_sig = 0
    for image in images:
        # append the image
        orignal_sig = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Remove black border by cropping
        h = orignal_sig.shape[0]
        w = orignal_sig.shape[1]
        y = 2
        x = 2
        crop_image_sig = orignal_sig[y:h - y, x:w - x]
        # rescale to 28 by 28 by 1
        r_w2_sig = 256.0 / crop_image_sig.shape[1]
        r_h2_sig = 256.0 / crop_image_sig.shape[0]
        dim_sig = (int(crop_image_sig.shape[1] * r_w2_sig), int(crop_image_sig.shape[0] * r_h2_sig))  # both
        # Format Image
        sig_resized_image = cv2.resize(crop_image_sig, dim_sig, interpolation=cv2.INTER_LINEAR)
        # Save the image
        cv2.imwrite(path_to_save + "\signature_prepared.png", sig_resized_image)

    return True