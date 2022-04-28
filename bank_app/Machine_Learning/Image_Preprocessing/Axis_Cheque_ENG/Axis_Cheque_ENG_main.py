from cv2 import cv2

from Pre_processing_Axis import preprocessing_before_crop_Axis
from word_preprocessing_Axis import word_cropping
from amount_preprocsessing_Axis import amount_cropping
from Date_Preprocessing_Axis import date_cropping
from ..Handwritten_Digits_Preparation.amount_digits_preprocessing import load_images_from_folder,Prepare_amount_Digits
from ..Handwritten_Digits_Preparation.date_digits_preprocessing import Prepare_Date_Digits
from signature_AXIS import signature_extraction_AXIS
from ..signature_preparation.signature_preparation import Prepare_signature
from ...Models.Words_Model_SimpleHTR.src.main import mainHTR
from ...Models.Handwritten_Digits_Model.handwritten_Amount_digits_Model_testing import predicted_amount, load_images_from_folder_amount
from ...Models.Handwritten_Digits_Model.handwritten_Date_digits_Model_testing import predicted_date, load_images_from_folder_date
from ...Models.Signature.signature_Model_testing import load_images_from_folder_signature, predicted_signature
import os

def Delete_all_files(path):
    for f in os.listdir(path):
        os.remove(os.path.join(path, f))
    return True



def main_AXIS_ENG():
    """Main function."""
    path_to_image = r"C:\Users\ADEM\PycharmProjects\Automatic_bank_cheques_data_extraction\static\saved_check_img_folder\uploaded_check.jpg"
    path_to_amount_folder = r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\segmented amount"
    path_to_date_folder = r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\segmented date"
    path_to_save_signature = r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\signature image AXIS"
    path_to_words_image = r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\segmented words"
    path_to_folder_amount = r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\Handwritten_Digits_Preparation\prepared_amount_digits_images"
    path_to_folder_date = r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\Handwritten_Digits_Preparation\prepared_date_digits_images"

    """Image preprocessing"""
    #path_to_image should take the input axis image cheque uploaded by the user

    orignal_image = cv2.cvtColor(cv2.imread(path_to_image), cv2.COLOR_BGR2RGB)
    #orignal_image should be assigned to an image field on the browser so that the user can see his uploaded image

    cropped_image = preprocessing_before_crop_Axis(path_to_image)
    #cropped_image should then replace the orignal_image on the same image field after 1 sec wait time

    words_image = word_cropping(cropped_image)
    #words_image should then be placed in the words_image_field after 1 sec wait time

    amount_image = amount_cropping(cropped_image)
    #amount_image should then be placed in the amount_image_field after 1 sec wait time

    """preparing AMOUNT digits image for prediction."""
    amount_digits_image = load_images_from_folder(path_to_amount_folder)
    Prepare_amount_Digits(amount_digits_image)

    date_image = date_cropping(cropped_image)
    #date_image should then be placed in the date_image_field after 1 sec wait time

    """preparing DATE digits image for prediction."""
    date_digits_image = load_images_from_folder(path_to_date_folder)
    Prepare_Date_Digits(date_digits_image)

    signature_image = signature_extraction_AXIS(path_to_image)
    #signature_image should then be placed in the signature_image_field after 1 sec wait time

    """preparing Signature image for prediction."""
    Prepare_signature(signature_image,path_to_save_signature)



    """ Prediction !!! """

    dict_of_predicted_words_nd_proba = mainHTR(path_to_words_image)
    """ for each key in dict_of_predicted_words_nd_proba 
    create a new text field and assign the key to that text field
    """
    """ for each value in dict_of_predicted_words_nd_proba 
        create a new text field and assign the value to that text field
    """
    To_predict_images_amount = []
    To_predict_images_amount = load_images_from_folder_amount(path_to_folder_amount)
    predicted_amount_str = predicted_amount(To_predict_images_amount)
    #predicted_amount_str should then be placed in the predicted_amount_str_field

    To_predict_images_date = []
    To_predict_images_date = load_images_from_folder_date(path_to_folder_date)
    predicted_date_str = predicted_date(To_predict_images_date)
    #predicted_date_str should then be placed in the predicted_date_str_field

    To_predict_images_signature = []
    To_predict_images_signature = load_images_from_folder_signature(path_to_save_signature)
    predicted_signature_str = predicted_signature(To_predict_images_signature)
    #predicted_signature_str should then be placed in the predicted_signature_str_str_field

    """ Deleting all the images for future tests """
    Delete_all_files(path_to_amount_folder)
    Delete_all_files(path_to_date_folder)
    Delete_all_files(path_to_save_signature)
    Delete_all_files(path_to_words_image)
    Delete_all_files(path_to_folder_amount)
    Delete_all_files(path_to_folder_date)










if __name__ == '__main_AXIS_ENG__':
    main_AXIS_ENG()