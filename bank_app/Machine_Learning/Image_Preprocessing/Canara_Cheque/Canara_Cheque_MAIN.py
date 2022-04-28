from Pre_processing_CANARA import preprocessing_before_crop_CANARA
from word_preprocessing_CANARA import word_cropping_CANARA
from amount_preprocsessing_CANARA import amount_cropping_CANARA
from Date_Preprocessing_CANARA import date_cropping_CANARA
from ..Handwritten_Digits_Preparation.amount_digits_preprocessing import load_images_from_folder,Prepare_amount_Digits
from ..Handwritten_Digits_Preparation.date_digits_preprocessing import Prepare_Date_Digits
from signature_CANARA import signature_extraction_CANARA
from ..signature_preparation.signature_preparation import Prepare_signature
from ...Models.Words_Model_SimpleHTR.src.main import mainHTR
from ...Models.Handwritten_Digits_Model.handwritten_Amount_digits_Model_testing import predicted_amount, load_images_from_folder_amount
from ...Models.Handwritten_Digits_Model.handwritten_Date_digits_Model_testing import predicted_date, load_images_from_folder_date
from ...Models.Signature.signature_Model_testing import load_images_from_folder_signature, predicted_signature
import cv2
import os

def Delete_all_files(path):
    for f in os.listdir(path):
        os.remove(os.path.join(path, f))
    return True



def main_CANARA_ENG():
    """Main function."""
    path_to_image_CANARA = r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\CANARA cheque\CANARA 2.png"
    path_to_amount_folder_CANARA = r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\CANARA cheque\CANARA amount images"
    path_to_date_folder_CANARA = r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\CANARA cheque\CANARA date images"
    path_to_save_signature_CANARA = r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\CANARA cheque\signature image CANARA"
    path_to_words_image_CANARA = r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\CANARA cheque\CANARA words images"
    path_to_folder_amount = r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\Handwritten_Digits_Preparation\prepared_amount_digits_images"
    path_to_folder_date = r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\Handwritten_Digits_Preparation\prepared_date_digits_images"

    """Image preprocessing"""
    #path_to_image should take the input axis image cheque uploaded by the user

    orignal_image = cv2.cvtColor(cv2.imread(path_to_image_CANARA), cv2.COLOR_BGR2RGB)
    #orignal_image should be assigned to an image field on the browser so that the user can see his uploaded image

    cropped_image = preprocessing_before_crop_CANARA(path_to_image_CANARA)
    #cropped_image should then replace the orignal_image on the same image field after 1 sec wait time

    words_image = word_cropping_CANARA(cropped_image)
    #words_image should then be placed in the words_image_field after 1 sec wait time

    amount_image = amount_cropping_CANARA(cropped_image)
    #amount_image should then be placed in the amount_image_field after 1 sec wait time

    """preparing AMOUNT digits image for prediction."""
    amount_digits_image = load_images_from_folder(path_to_amount_folder_CANARA)
    Prepare_amount_Digits(amount_digits_image)

    date_image = date_cropping_CANARA(cropped_image)
    #date_image should then be placed in the date_image_field after 1 sec wait time

    """preparing DATE digits image for prediction."""
    date_digits_image = load_images_from_folder(path_to_date_folder_CANARA)
    Prepare_Date_Digits(date_digits_image)

    signature_image = signature_extraction_CANARA(path_to_image_CANARA)
    #signature_image should then be placed in the signature_image_field after 1 sec wait time

    """preparing Signature image for prediction."""
    Prepare_signature(signature_image,path_to_save_signature_CANARA)



    """ Prediction !!! """

    dict_of_predicted_words_nd_proba = mainHTR(path_to_words_image_CANARA)
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
    To_predict_images_signature = load_images_from_folder_signature(path_to_save_signature_CANARA)
    predicted_signature_str = predicted_signature(To_predict_images_signature)
    #predicted_signature_str should then be placed in the predicted_signature_str_str_field

    """ Deleting all the images for future tests """
    Delete_all_files(path_to_amount_folder_CANARA)
    Delete_all_files(path_to_date_folder_CANARA)
    Delete_all_files(path_to_save_signature_CANARA)
    Delete_all_files(path_to_words_image_CANARA)
    Delete_all_files(path_to_folder_amount)
    Delete_all_files(path_to_folder_date)










if __name__ == '__main_CANARA_ENG__':
    main_CANARA_ENG()