import os
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
    #img = img[:,:,0]
    # reshape into a single sample with 1 channel
    #img = img.reshape(1, 256, 256, 1)
    return img_batch

def load_images_from_folder_arabic(folder):
    images = []
    for filename in os.listdir(folder):
        img = load_image(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

#Test
path_to_words_folder_arabic = r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\Arabic cheques\segmented words arabic"
To_predict_images_arabic = []
To_predict_images_arabic = load_images_from_folder_arabic(path_to_words_folder_arabic)
def predicted_arabic_words(To_predict_images_arabic):
    # -------------------- to avoid Tensorflow eager mode --------------------#
    tf.compat.v1.disable_eager_execution()
    # ------------------------------------------------------------------------#

    dictionary_words = {0: "آدم", 1: "آلاف", 2: "خمسة", 3: "مائة", 4: "و", 5: "يوسف"}
    # dictionary_words2 = {0: "adem", 1: "alef", 2: "khamsatou", 3: "miatou", 4: "waa", 5: "youssef"}
    words_str = ""
    for image in reversed(To_predict_images_arabic):
        # load model
        model = load_model(r'C:\Users\ADEM\PycharmProjects\Automatic_bank_cheques_data_extraction\bank_app\Machine_Learning\Models\Arabic_Words_Model\final_Handwritten_words_arabic_model.h5')
        # predict the class
        predict_word = model.predict(image)
        word = argmax(predict_word)
        word = dictionary_words[word]
        words_str += str(word) + ' '
    # transforming the output of the prediction to an amount of money

    # final_string = ''.join(words_str)
    return words_str

