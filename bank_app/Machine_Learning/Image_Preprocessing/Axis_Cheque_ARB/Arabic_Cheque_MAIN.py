from Pre_processing_arabic import preprocessing_before_crop_Axis_arabic
from word_preprocessing_arabic import word_cropping_arabic
from amount_preprocsessing_Axis_arabic import amount_cropping_arabic
from Date_Preprocessing_arabic import date_cropping_arabic
from ..Handwritten_Digits_Preparation.amount_digits_preprocessing import load_images_from_folder,Prepare_amount_Digits
from ..Handwritten_Digits_Preparation.date_digits_preprocessing import Prepare_Date_Digits
from signature_arabic import signature_extraction_AXIS_arabic
from ..signature_preparation.signature_preparation import Prepare_signature
from ...Models.Arabic_Words_Model.handwritten_arabic_words_Model_testing import predicted_arabic_words, load_images_from_folder_arabic
from ...Models.Handwritten_Digits_Model.handwritten_Amount_digits_Model_testing import predicted_amount, load_images_from_folder_amount
from ...Models.Handwritten_Digits_Model.handwritten_Date_digits_Model_testing import predicted_date, load_images_from_folder_date
from ...Models.Signature.signature_Model_testing import load_images_from_folder_signature, predicted_signature
import cv2
import os

def Delete_all_files(path):
    for f in os.listdir(path):
        os.remove(os.path.join(path, f))
    return True



def main_Arabic_ARB():
    """Main function."""
    path_to_image_ARB = r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\Arabic cheques\fix4.jpg"
    path_to_amount_folder_ARB = r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\Arabic cheques\segmented amount arabic"
    path_to_date_folder_ARB = r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\Arabic cheques\segmented date arabic"
    path_to_save_signature_ARB = r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\Arabic cheques\signature image AXIS arabic"
    path_to_words_image_ARB = r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\Arabic cheques\segmented words arabic"
    path_to_folder_amount = r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\Handwritten_Digits_Preparation\prepared_amount_digits_images"
    path_to_folder_date = r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\Handwritten_Digits_Preparation\prepared_date_digits_images"

    """Image preprocessing"""
    #path_to_image should take the input axis image cheque uploaded by the user

    orignal_image = cv2.cvtColor(cv2.imread(path_to_image_ARB), cv2.COLOR_BGR2RGB)
    #orignal_image should be assigned to an image field on the browser so that the user can see his uploaded image

    cropped_image = preprocessing_before_crop_Axis_arabic(path_to_image_ARB)
    #cropped_image should then replace the orignal_image on the same image field after 1 sec wait time

    words_image = word_cropping_arabic(cropped_image)
    #words_image should then be placed in the words_image_field after 1 sec wait time

    amount_image = amount_cropping_arabic(cropped_image)
    #amount_image should then be placed in the amount_image_field after 1 sec wait time

    """preparing AMOUNT digits image for prediction."""
    amount_digits_image = load_images_from_folder(path_to_amount_folder_ARB)
    Prepare_amount_Digits(amount_digits_image)

    date_image = date_cropping_arabic(cropped_image)
    #date_image should then be placed in the date_image_field after 1 sec wait time

    """preparing DATE digits image for prediction."""
    date_digits_image = load_images_from_folder(path_to_date_folder_ARB)
    Prepare_Date_Digits(date_digits_image)

    signature_image = signature_extraction_AXIS_arabic(path_to_image_ARB)
    #signature_image should then be placed in the signature_image_field after 1 sec wait time

    """preparing Signature image for prediction."""
    Prepare_signature(signature_image,path_to_save_signature_ARB)



    """ Prediction !!! """
    To_predict_images_words = []
    To_predict_images_words = load_images_from_folder_arabic(path_to_words_image_ARB)
    predicted_words_str = predicted_arabic_words(To_predict_images_words)
    """ 
    just take the predicted_words_str and put it in the text field of the words
    for the probabilities we can put "Under Dev"
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
    To_predict_images_signature = load_images_from_folder_signature(path_to_save_signature_ARB)
    predicted_signature_str = predicted_signature(To_predict_images_signature)
    #predicted_signature_str should then be placed in the predicted_signature_str_str_field

    """ Deleting all the images for future tests """
    Delete_all_files(path_to_amount_folder_ARB)
    Delete_all_files(path_to_date_folder_ARB)
    Delete_all_files(path_to_save_signature_ARB)
    Delete_all_files(path_to_words_image_ARB)
    Delete_all_files(path_to_folder_amount)
    Delete_all_files(path_to_folder_date)










if __name__ == '__main_Arabic_ARB__':
    main_Arabic_ARB()