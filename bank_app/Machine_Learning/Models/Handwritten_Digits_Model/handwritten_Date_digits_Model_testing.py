import os
from numpy import argmax
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

#Prepare image for prediction
def load_image(filename):
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
def load_images_from_folder_date(folder):
    images = []
    for filename in os.listdir(folder):
        img = load_image(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images


#Test
path_to_folder_date = r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\Handwritten_Digits_Preparation\prepared_date_digits_images"
To_predict_images_date = []
To_predict_images_date = load_images_from_folder_date(path_to_folder_date)


def predicted_date(To_predict_images_date):
    # -------------------- to avoid Tensorflow eager mode --------------------#
    tf.compat.v1.disable_eager_execution()
    # ------------------------------------------------------------------------#
    digit_str = ""
    for image in To_predict_images_date:
        # load model
        model = load_model(r'C:\Users\ADEM\PycharmProjects\Automatic_bank_cheques_data_extraction\bank_app\Machine_Learning\Models\Handwritten_Digits_Model\final_Handwritten_digits_model.h5')
        # predict the class
        predict_value = model.predict(image)
        # predict_proba = model.predict_proba(image)
        digit = argmax(predict_value)
        # digit_proba = argmax(predict_proba)
        digit_str += str(digit)
        # print(digit_proba)

    digit_str = list(digit_str)
    digit_str.insert(2, '/')
    digit_str.insert(5, '/')
    final_string = ''.join(digit_str)
    return final_string
