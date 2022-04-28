import os
import cv2
from numpy import argmax
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np

#Load all images
def load_image(filename):
    # load the image
    img = load_img(filename, grayscale=False, target_size=(256, 256))
    # convert to array
    img = img_to_array(img)
    img_batch = np.expand_dims(img, axis=0)
    return img_batch


def load_images_from_folder_signature(folder):
    images = []
    for filename in os.listdir(folder):
        if filename == r"signature_prepared.png":
            img = load_image(os.path.join(folder, filename))
            if img is not None:
                images.append(img)

    return images

#Test
path_to_sig_folder = r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\CANARA cheque\signature image CANARA"
To_predict_images_signature = []
To_predict_images_signature = load_images_from_folder_signature(path_to_sig_folder)

def predicted_signature(To_predict_images_signature):
    # -------------------- to avoid Tensorflow eager mode --------------------#
    tf.compat.v1.disable_eager_execution()
    # ------------------------------------------------------------------------#
    dictionary_signature = {0: "sig_axis_ENG_condidate", 1: "sig_adem_Axis_ARB", 2: "sig_Canara_Bank_condidate", 3: "sig_Syndicate_condidate"}
    sig_str = ""
    for image in To_predict_images_signature:
        # load model
        model = load_model(r'C:\Users\ADEM\PycharmProjects\Automatic_bank_cheques_data_extraction\bank_app\Machine_Learning\Models\Signature\final_signature_model.h5')
        # predict the class
        predict_sig = model.predict(image)
        sig = argmax(predict_sig)
        sig = dictionary_signature[sig]
        sig_str += str(sig)
    # transforming the output of the prediction to an amount of money

    # final_string = ''.join(words_str)
    return sig_str
