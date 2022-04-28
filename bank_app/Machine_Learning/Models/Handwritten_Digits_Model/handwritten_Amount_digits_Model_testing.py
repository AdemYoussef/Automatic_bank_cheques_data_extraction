import os
import cv2
from numpy import argmax
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np


#Prepare image for prediction
def load_image_f(filename):
    # load the image
    img = load_img(filename, grayscale=True, target_size=(28, 28))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 1 channel
    img = img.reshape(1, 28, 28, 1)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img

#Load all images
def load_images_from_folder_amount(folder):
    images = []
    for filename in os.listdir(folder):
        img = load_image_f(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

#Test
path_to_folder_amount = r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\Handwritten_Digits_Preparation\prepared_amount_digits_images"
To_predict_images = []
To_predict_images = load_images_from_folder_amount(path_to_folder_amount)


def predicted_amount(To_predict_images):
    digit_str = ""
    # -------------------- to avoid Tensorflow eager mode --------------------#
    tf.compat.v1.disable_eager_execution()
    # ------------------------------------------------------------------------#
    for image in To_predict_images:
        # load model
        model = load_model(r'C:\Users\ADEM\PycharmProjects\Automatic_bank_cheques_data_extraction\bank_app\Machine_Learning\Models\Handwritten_Digits_Model\final_Handwritten_digits_model.h5')
        # predict the class
        predict_value = model.predict(image)
        digit = argmax(predict_value)
        digit_str += str(digit)
    # transforming the output of the prediction to an amount of money
    digit_str = list(digit_str)
    final_string = ''.join(digit_str)
    return final_string
    # to remove the currency we just remove the first prediction
    # final_string = final_string[1:]


#print(final_string)