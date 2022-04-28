import os
import subprocess
import sys

import cv2

from django.shortcuts import render, redirect
from django.http import HttpRequest, HttpResponse

from .Machine_Learning.Image_Preprocessing.Axis_Cheque_ENG.Date_Preprocessing_Axis import date_cropping
from .Machine_Learning.Image_Preprocessing.Axis_Cheque_ENG.Pre_processing_Axis import preprocessing_before_crop_Axis
from .Machine_Learning.Image_Preprocessing.Axis_Cheque_ENG.amount_preprocsessing_Axis import amount_cropping
from .Machine_Learning.Image_Preprocessing.Axis_Cheque_ENG.signature_AXIS import signature_extraction_AXIS
from .Machine_Learning.Image_Preprocessing.Axis_Cheque_ENG.word_preprocessing_Axis import word_cropping

from .Machine_Learning.Image_Preprocessing.Canara_Cheque.Date_Preprocessing_CANARA import date_cropping_CANARA
from .Machine_Learning.Image_Preprocessing.Canara_Cheque.Pre_processing_CANARA import preprocessing_before_crop_CANARA
from .Machine_Learning.Image_Preprocessing.Canara_Cheque.amount_preprocsessing_CANARA import amount_cropping_CANARA
from .Machine_Learning.Image_Preprocessing.Canara_Cheque.signature_CANARA import signature_extraction_CANARA
from .Machine_Learning.Image_Preprocessing.Canara_Cheque.word_preprocessing_CANARA import word_cropping_CANARA

from .Machine_Learning.Image_Preprocessing.Syndicate_Cheque.Date_Preprocessing_kelb import date_cropping_kelb
from .Machine_Learning.Image_Preprocessing.Syndicate_Cheque.Pre_processing_kelb import preprocessing_before_crop_kelb
from .Machine_Learning.Image_Preprocessing.Syndicate_Cheque.amount_preprocsessing_kelb import amount_cropping_kelb
from .Machine_Learning.Image_Preprocessing.Syndicate_Cheque.signature_KELB import signature_extraction_KELB
from .Machine_Learning.Image_Preprocessing.Syndicate_Cheque.word_preprocessing_kelb import word_cropping_kelb

from .Machine_Learning.Image_Preprocessing.Axis_Cheque_ARB.Date_Preprocessing_arabic import date_cropping_arabic
from .Machine_Learning.Image_Preprocessing.Axis_Cheque_ARB.Pre_processing_arabic import preprocessing_before_crop_Axis_arabic
from .Machine_Learning.Image_Preprocessing.Axis_Cheque_ARB.amount_preprocsessing_Axis_arabic import amount_cropping_arabic
from .Machine_Learning.Image_Preprocessing.Axis_Cheque_ARB.signature_arabic import signature_extraction_AXIS_arabic
from .Machine_Learning.Image_Preprocessing.Axis_Cheque_ARB.word_preprocessing_arabic import word_cropping_arabic

from .Machine_Learning.Image_Preprocessing.Handwritten_Digits_Preparation.amount_digits_preprocessing import load_images_from_folder,Prepare_amount_Digits
from .Machine_Learning.Image_Preprocessing.Handwritten_Digits_Preparation.date_digits_preprocessing import \
    Prepare_Date_Digits
from .Machine_Learning.Image_Preprocessing.arabic_preparation.arabic_words_preparation import Prepare_arabic_words
from .Machine_Learning.Image_Preprocessing.signature_preparation.signature_preparation import Prepare_signature
from .Machine_Learning.Models.Arabic_Words_Model.handwritten_arabic_words_Model_testing import \
    load_images_from_folder_arabic, predicted_arabic_words
from .Machine_Learning.Models.Handwritten_Digits_Model.handwritten_Amount_digits_Model_testing import \
    load_images_from_folder_amount, predicted_amount
from .Machine_Learning.Models.Handwritten_Digits_Model.handwritten_Date_digits_Model_testing import \
    load_images_from_folder_date, predicted_date
from .Machine_Learning.Models.Signature.signature_Model_testing import load_images_from_folder_signature, \
    predicted_signature


def index(request):
    return render(request, 'index.html')

def loginView(request):
    return render(request,'login.html')

def signIn(request):
    return render(request, 'index.html')

def notFound(request):
    return render(request, '404.html')

def Delete_all_files(path):
    for f in os.listdir(path):
        os.remove(os.path.join(path, f))
    return True

def advanced_form_components(request):
    path_to_image = r"C:\Users\ADEM\PycharmProjects\Automatic_bank_cheques_data_extraction\static\saved_check_img_folder\uploaded_check.jpg"
    path_to_folder_amount = r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\Handwritten_Digits_Preparation\prepared_amount_digits_images"
    path_to_folder_date = r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\Handwritten_Digits_Preparation\prepared_date_digits_images"
    path_to_static_imgs = r'C:\Users\ADEM\PycharmProjects\Automatic_bank_cheques_data_extraction\static\img\image_preprocessing'
    '''---AXIS-ENG---'''
    path_to_amount_folder_AXIS = r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\segmented amount"
    path_to_date_folder_AXIS = r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\segmented date"
    path_to_save_signature_AXIS = r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\signature image AXIS"
    path_to_words_image_AXIS = r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\segmented words"

    '''---end AXIS-ENG---'''
    '''---CANARA---'''
    path_to_amount_folder_CANARA = r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\CANARA cheque\CANARA amount images"
    path_to_date_folder_CANARA = r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\CANARA cheque\CANARA date images"
    path_to_save_signature_CANARA = r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\CANARA cheque\signature image CANARA"
    path_to_words_image_CANARA = r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\CANARA cheque\CANARA words images"

    '''---end CANARA---'''
    '''---Syndicate---'''
    path_to_amount_folder_Syndicate = r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\kelb cheque\kelb amount images"
    path_to_date_folder_Syndicate = r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\kelb cheque\kelb date images"
    path_to_save_signature_Syndicate = r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\kelb cheque\signature image KELB"
    path_to_words_image_Syndicate = r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\kelb cheque\kelb word images"

    '''---end CANARA---'''

    '''---Axis Arabic---'''
    path_to_amount_folder_AXIS_ARB = r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\Arabic cheques\segmented amount arabic"
    path_to_date_folder_AXIS_ARB = r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\Arabic cheques\segmented date arabic"
    path_to_save_signature_AXIS_ARB = r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\Arabic cheques\signature image AXIS arabic"
    path_to_words_image_AXIS_ARB = r"C:\Users\ADEM\Desktop\ESPRIT_Education\4er\PI DS\image preprocessing\Arabic cheques\segmented words arabic"

    '''---end CANARA---'''



    if request.method == 'POST':
        img = request.FILES['uploaded_check']
        #print(dir(img))
        img_name = img.name
        img_extension = os.path.splitext(img_name)[1]  # 1
        saved_check_img_folder = r'C:/Users/ADEM/PycharmProjects/Automatic_bank_cheques_data_extraction/static/saved_check_img_folder'
        if not os.path.exists(saved_check_img_folder):
            os.mkdir(saved_check_img_folder)
        img_save_path = saved_check_img_folder + '/uploaded_check' + img_extension  # 2
        with open(img_save_path, 'wb+') as destination:
            for chunk in img.chunks():
                destination.write(chunk)
        data = request.POST
        action_check_language = data.get("check_language")
        action_check_bank = data.get("check_bank")
        if(action_check_bank == "AXIS" and action_check_language == "English" ):
            request.session['bank_name'] = action_check_bank
            request.session['bank_language'] = action_check_language
            request.session['predicted_words'] = ''
            request.session['predicted_proba'] = ''
            request.session['predicted_amount'] = ''
            request.session['predicted_date'] = ''
            request.session['predicted_sig'] = ''

            cropped_image = preprocessing_before_crop_Axis(path_to_image)
            words_image = word_cropping(cropped_image)
            cv2.imwrite(r'C:\Users\ADEM\PycharmProjects\Automatic_bank_cheques_data_extraction\static\img\image_preprocessing\words_image.png', words_image)
            amount_image = amount_cropping(cropped_image)
            cv2.imwrite(r'C:\Users\ADEM\PycharmProjects\Automatic_bank_cheques_data_extraction\static\img\image_preprocessing\amount_image.png', amount_image)
            """preparing AMOUNT digits image for prediction."""
            amount_digits_image = load_images_from_folder(path_to_amount_folder_AXIS)
            Prepare_amount_Digits(amount_digits_image)
            date_image = date_cropping(cropped_image)
            cv2.imwrite(r'C:\Users\ADEM\PycharmProjects\Automatic_bank_cheques_data_extraction\static\img\image_preprocessing\date_image.png', date_image)
            """preparing DATE digits image for prediction."""
            date_digits_image = load_images_from_folder(path_to_date_folder_AXIS)
            Prepare_Date_Digits(date_digits_image)
            signature_image = signature_extraction_AXIS(path_to_image)
            cv2.imwrite(r'C:\Users\ADEM\PycharmProjects\Automatic_bank_cheques_data_extraction\static\img\image_preprocessing\signature_image.png', signature_image)
            """preparing Signature image for prediction."""
            Prepare_signature(load_images_from_folder(path_to_save_signature_AXIS), path_to_save_signature_AXIS)
            #Prepare_signature(signature_image, path_to_save_signature_AXIS)
            """ Prediction !!! """

            #dict_of_predicted_words_nd_proba = subprocess.run(['python', "C:/Users/ADEM/Desktop/d_with_word/SimpleHTR_master/src/main.py", '--imgs_folder', "C:/Users/ADEM/Desktop/ESPRIT_Education/4er/PI DS/image preprocessing/segmented words"],capture_output=True)
            dict_of_predicted_words_nd_proba = subprocess.run(args=['python', "C:/Users/ADEM/Desktop/d_with_word/SimpleHTR_master/src/main.py", '--imgs_folder', "C:/Users/ADEM/Desktop/ESPRIT_Education/4er/PI DS/image preprocessing/segmented words"],
                             universal_newlines=True,
                             stdout=subprocess.PIPE)
            dict_of_predicted_words_nd_proba = dict_of_predicted_words_nd_proba.stdout.splitlines()
            dict_of_predicted_words_nd_proba = str(dict_of_predicted_words_nd_proba[-1])
            predicted_full_name_and_amount_phrase = ''
            predicted_full_proba_name_and_amount_phrase = ''
            full_predicted_str = str(dict_of_predicted_words_nd_proba)
            full_predicted_str = full_predicted_str.split(' ',2)
            name_str = full_predicted_str[0] + ' ' + full_predicted_str[1]
            amount_str = full_predicted_str[2]
            request.session['name_str'] = name_str
            request.session['amount_str'] = amount_str
            #request.session['predicted_proba'] = '----Under Development----'


            To_predict_images_amount = []
            To_predict_images_amount = load_images_from_folder_amount(path_to_folder_amount)
            predicted_amount_str = predicted_amount(To_predict_images_amount)
            request.session['predicted_amount'] = predicted_amount_str

            To_predict_images_date = []
            To_predict_images_date = load_images_from_folder_date(path_to_folder_date)
            predicted_date_str = predicted_date(To_predict_images_date)
            request.session['predicted_date'] = predicted_date_str

            To_predict_images_signature = []
            To_predict_images_signature = load_images_from_folder_signature(path_to_save_signature_AXIS)
            predicted_signature_str = predicted_signature(To_predict_images_signature)
            request.session['predicted_sig'] = predicted_signature_str

            """ Deleting all the images for future tests """
            Delete_all_files(path_to_amount_folder_AXIS)
            Delete_all_files(path_to_date_folder_AXIS)
            Delete_all_files(path_to_save_signature_AXIS)
            Delete_all_files(path_to_words_image_AXIS)
            Delete_all_files(path_to_folder_amount)
            Delete_all_files(path_to_folder_date)


            return redirect('/form_validation', {'bank_name': action_check_bank},{'bank_language': action_check_language},
                            {'name_str': name_str}
                            , {'amount_str': amount_str},
                            {'predicted_proba':predicted_full_proba_name_and_amount_phrase}
                            ,{'predicted_amount':predicted_amount_str},{'predicted_date':predicted_date_str}
                            ,{'predicted_sig':predicted_signature_str})

        elif( action_check_bank == "CANARA" and action_check_language == "English" ):

            request.session['bank_name'] = action_check_bank
            request.session['bank_language'] = action_check_language
            request.session['predicted_words'] = ''
            request.session['predicted_proba'] = ''
            request.session['predicted_amount'] = ''
            request.session['predicted_date'] = ''
            request.session['predicted_sig'] = ''

            cropped_image = preprocessing_before_crop_CANARA(path_to_image)
            words_image = word_cropping_CANARA(cropped_image)
            cv2.imwrite(r'C:\Users\ADEM\PycharmProjects\Automatic_bank_cheques_data_extraction\static\img\image_preprocessing\words_image.png', words_image)
            amount_image = amount_cropping_CANARA(cropped_image)
            cv2.imwrite(r'C:\Users\ADEM\PycharmProjects\Automatic_bank_cheques_data_extraction\static\img\image_preprocessing\amount_image.png', amount_image)
            """preparing AMOUNT digits image for prediction."""
            amount_digits_image = load_images_from_folder(path_to_amount_folder_CANARA)
            Prepare_amount_Digits(amount_digits_image)
            date_image = date_cropping_CANARA(cropped_image)
            cv2.imwrite(r'C:\Users\ADEM\PycharmProjects\Automatic_bank_cheques_data_extraction\static\img\image_preprocessing\date_image.png', date_image)
            """preparing DATE digits image for prediction."""
            date_digits_image = load_images_from_folder(path_to_date_folder_CANARA)
            Prepare_Date_Digits(date_digits_image)
            signature_image = signature_extraction_CANARA(path_to_image)
            cv2.imwrite(r'C:\Users\ADEM\PycharmProjects\Automatic_bank_cheques_data_extraction\static\img\image_preprocessing\signature_image.png', signature_image)
            """preparing Signature image for prediction."""
            Prepare_signature(load_images_from_folder(path_to_save_signature_CANARA), path_to_save_signature_CANARA)
            # Prepare_signature(signature_image, path_to_save_signature_AXIS)
            """ Prediction !!! """

            dict_of_predicted_words_nd_proba = subprocess.run(
                args=['python', "C:/Users/ADEM/Desktop/d_with_word/SimpleHTR_master/src/main.py", '--imgs_folder',
                      "C:/Users/ADEM/Desktop/ESPRIT_Education/4er/PI DS/image preprocessing/CANARA cheque/CANARA words images"],
                universal_newlines=True,
                stdout=subprocess.PIPE)
            dict_of_predicted_words_nd_proba = dict_of_predicted_words_nd_proba.stdout.splitlines()
            dict_of_predicted_words_nd_proba = str(dict_of_predicted_words_nd_proba[-1])
            predicted_full_name_and_amount_phrase = ''
            predicted_full_proba_name_and_amount_phrase = ''
            full_predicted_str = str(dict_of_predicted_words_nd_proba)
            full_predicted_str = full_predicted_str.split(' ', 2)
            name_str = full_predicted_str[0] + ' ' + full_predicted_str[1]
            amount_str = full_predicted_str[2]
            request.session['name_str'] = name_str
            request.session['amount_str'] = amount_str
            #request.session['predicted_words'] = str(dict_of_predicted_words_nd_proba)
            # request.session['predicted_proba'] = '----Under Development----'

            To_predict_images_amount = []
            To_predict_images_amount = load_images_from_folder_amount(path_to_folder_amount)
            predicted_amount_str = predicted_amount(To_predict_images_amount)
            request.session['predicted_amount'] = predicted_amount_str

            To_predict_images_date = []
            To_predict_images_date = load_images_from_folder_date(path_to_folder_date)
            predicted_date_str = predicted_date(To_predict_images_date)
            request.session['predicted_date'] = predicted_date_str

            To_predict_images_signature = []
            To_predict_images_signature = load_images_from_folder_signature(path_to_save_signature_CANARA)
            predicted_signature_str = predicted_signature(To_predict_images_signature)
            request.session['predicted_sig'] = predicted_signature_str

            """ Deleting all the images for future tests """
            Delete_all_files(path_to_amount_folder_CANARA)
            Delete_all_files(path_to_date_folder_CANARA)
            Delete_all_files(path_to_save_signature_CANARA)
            Delete_all_files(path_to_words_image_CANARA)
            Delete_all_files(path_to_folder_amount)
            Delete_all_files(path_to_folder_date)


            return redirect('/form_validation', {'bank_name': action_check_bank},
                            {'bank_language': action_check_language},
                            {'name_str': name_str}
                            , {'amount_str': amount_str},
                            {'predicted_proba': predicted_full_proba_name_and_amount_phrase}
                            , {'predicted_amount': predicted_amount_str}, {'predicted_date': predicted_date_str}
                            , {'predicted_sig': predicted_signature_str})
        elif (action_check_bank == "Syndicate" and action_check_language == "English"):

            request.session['bank_name'] = action_check_bank
            request.session['bank_language'] = action_check_language
            request.session['predicted_words'] = ''
            request.session['predicted_proba'] = ''
            request.session['predicted_amount'] = ''
            request.session['predicted_date'] = ''
            request.session['predicted_sig'] = ''

            cropped_image = preprocessing_before_crop_kelb(path_to_image)
            words_image = word_cropping_kelb(cropped_image)
            cv2.imwrite(r'C:\Users\ADEM\PycharmProjects\Automatic_bank_cheques_data_extraction\static\img\image_preprocessing\words_image.png', words_image)
            amount_image = amount_cropping_kelb(cropped_image)
            cv2.imwrite(r'C:\Users\ADEM\PycharmProjects\Automatic_bank_cheques_data_extraction\static\img\image_preprocessing\amount_image.png', amount_image)
            """preparing AMOUNT digits image for prediction."""
            amount_digits_image = load_images_from_folder(path_to_amount_folder_Syndicate)
            Prepare_amount_Digits(amount_digits_image)
            date_image = date_cropping_kelb(cropped_image)
            cv2.imwrite(r'C:\Users\ADEM\PycharmProjects\Automatic_bank_cheques_data_extraction\static\img\image_preprocessing\date_image.png', date_image)
            """preparing DATE digits image for prediction."""
            date_digits_image = load_images_from_folder(path_to_date_folder_Syndicate)
            Prepare_Date_Digits(date_digits_image)
            signature_image = signature_extraction_KELB(path_to_image)
            cv2.imwrite(r'C:\Users\ADEM\PycharmProjects\Automatic_bank_cheques_data_extraction\static\img\image_preprocessing\signature_image.png', signature_image)
            """preparing Signature image for prediction."""
            Prepare_signature(load_images_from_folder(path_to_save_signature_Syndicate), path_to_save_signature_Syndicate)
            # Prepare_signature(signature_image, path_to_save_signature_AXIS)
            """ Prediction !!! """

            dict_of_predicted_words_nd_proba = subprocess.run(
                args=['python', "C:/Users/ADEM/Desktop/d_with_word/SimpleHTR_master/src/main.py", '--imgs_folder',
                      "C:/Users/ADEM/Desktop/ESPRIT_Education/4er/PI DS/image preprocessing/kelb cheque/kelb word images"],
                universal_newlines=True,
                stdout=subprocess.PIPE)
            dict_of_predicted_words_nd_proba = dict_of_predicted_words_nd_proba.stdout.splitlines()
            dict_of_predicted_words_nd_proba = str(dict_of_predicted_words_nd_proba[-1])
            predicted_full_name_and_amount_phrase = ''
            predicted_full_proba_name_and_amount_phrase = ''
            full_predicted_str = str(dict_of_predicted_words_nd_proba)
            full_predicted_str = full_predicted_str.split(' ', 2)
            name_str = full_predicted_str[0] + ' ' + full_predicted_str[1]
            amount_str = full_predicted_str[2]
            request.session['name_str'] = name_str
            request.session['amount_str'] = amount_str
            #request.session['predicted_words'] = str(dict_of_predicted_words_nd_proba)
            # request.session['predicted_proba'] = '----Under Development----'

            To_predict_images_amount = []
            To_predict_images_amount = load_images_from_folder_amount(path_to_folder_amount)
            predicted_amount_str = predicted_amount(To_predict_images_amount)
            request.session['predicted_amount'] = predicted_amount_str

            To_predict_images_date = []
            To_predict_images_date = load_images_from_folder_date(path_to_folder_date)
            predicted_date_str = predicted_date(To_predict_images_date)
            request.session['predicted_date'] = predicted_date_str

            To_predict_images_signature = []
            To_predict_images_signature = load_images_from_folder_signature(path_to_save_signature_Syndicate)
            predicted_signature_str = predicted_signature(To_predict_images_signature)
            request.session['predicted_sig'] = predicted_signature_str

            """ Deleting all the images for future tests """
            Delete_all_files(path_to_amount_folder_Syndicate)
            Delete_all_files(path_to_date_folder_Syndicate)
            Delete_all_files(path_to_save_signature_Syndicate)
            Delete_all_files(path_to_words_image_Syndicate)
            Delete_all_files(path_to_folder_amount)
            Delete_all_files(path_to_folder_date)


            return redirect('/form_validation', {'bank_name': action_check_bank},
                            {'bank_language': action_check_language},
                            {'name_str': name_str}
                            , {'amount_str': amount_str},
                            {'predicted_proba': predicted_full_proba_name_and_amount_phrase}
                            , {'predicted_amount': predicted_amount_str}, {'predicted_date': predicted_date_str}
                            , {'predicted_sig': predicted_signature_str})
        else:
            request.session['bank_name'] = action_check_bank
            request.session['bank_language'] = action_check_language
            request.session['predicted_words'] = ''
            request.session['predicted_proba'] = ''
            request.session['predicted_amount'] = ''
            request.session['predicted_date'] = ''
            request.session['predicted_sig'] = ''

            cropped_image = preprocessing_before_crop_Axis_arabic(path_to_image)
            words_image = word_cropping_arabic(cropped_image)
            cv2.imwrite(r'C:\Users\ADEM\PycharmProjects\Automatic_bank_cheques_data_extraction\static\img\image_preprocessing\words_image.png', words_image)
            amount_image = amount_cropping_arabic(cropped_image)
            cv2.imwrite(r'C:\Users\ADEM\PycharmProjects\Automatic_bank_cheques_data_extraction\static\img\image_preprocessing\amount_image.png', amount_image)
            """preparing AMOUNT digits image for prediction."""
            amount_digits_image = load_images_from_folder(path_to_amount_folder_AXIS_ARB)
            Prepare_amount_Digits(amount_digits_image)
            date_image = date_cropping_arabic(cropped_image)
            cv2.imwrite(r'C:\Users\ADEM\PycharmProjects\Automatic_bank_cheques_data_extraction\static\img\image_preprocessing\date_image.png', date_image)
            """preparing DATE digits image for prediction."""
            date_digits_image = load_images_from_folder(path_to_date_folder_AXIS_ARB)
            Prepare_Date_Digits(date_digits_image)
            signature_image = signature_extraction_AXIS_arabic(path_to_image)
            cv2.imwrite(r'C:\Users\ADEM\PycharmProjects\Automatic_bank_cheques_data_extraction\static\img\image_preprocessing\signature_image.png', signature_image)
            """preparing Signature image for prediction."""
            Prepare_signature(load_images_from_folder(path_to_save_signature_AXIS_ARB),path_to_save_signature_AXIS_ARB)
            # Prepare_signature(signature_image, path_to_save_signature_AXIS)
            """preparing Arabic words image for prediction."""
            arabic_words_images = load_images_from_folder(path_to_words_image_AXIS_ARB)
            Prepare_arabic_words(arabic_words_images)
            """ Prediction !!! """

            To_predict_images_arabic = []
            To_predict_images_words_arabic = load_images_from_folder_arabic(path_to_words_image_AXIS_ARB)
            predicted_words_str = predicted_arabic_words(To_predict_images_words_arabic)
            full_predicted_str = predicted_words_str.split(' ', 2)
            name_str = full_predicted_str[0] + ' ' + full_predicted_str[1]
            amount_str = full_predicted_str[2]
            request.session['name_str'] = name_str
            request.session['amount_str'] = amount_str
            #request.session['predicted_words'] = predicted_words_str


            predicted_full_proba_name_and_amount_phrase = ''
            # request.session['predicted_proba'] = '----Under Development----'

            To_predict_images_amount = []
            To_predict_images_amount = load_images_from_folder_amount(path_to_folder_amount)
            predicted_amount_str = predicted_amount(To_predict_images_amount)
            request.session['predicted_amount'] = predicted_amount_str

            To_predict_images_date = []
            To_predict_images_date = load_images_from_folder_date(path_to_folder_date)
            predicted_date_str = predicted_date(To_predict_images_date)
            request.session['predicted_date'] = predicted_date_str

            To_predict_images_signature = []
            To_predict_images_signature = load_images_from_folder_signature(path_to_save_signature_AXIS_ARB)
            predicted_signature_str = predicted_signature(To_predict_images_signature)
            request.session['predicted_sig'] = predicted_signature_str

            """ Deleting all the images for future tests """
            Delete_all_files(path_to_amount_folder_AXIS_ARB)
            Delete_all_files(path_to_date_folder_AXIS_ARB)
            Delete_all_files(path_to_save_signature_Syndicate)
            Delete_all_files(path_to_words_image_AXIS_ARB)
            Delete_all_files(path_to_folder_amount)
            Delete_all_files(path_to_folder_date)


            return redirect('/form_validation', {'bank_name': action_check_bank},
                            {'bank_language': action_check_language}
                            , {'name_str': name_str}
                            , {'amount_str': amount_str},
                            {'predicted_proba': predicted_full_proba_name_and_amount_phrase}
                            , {'predicted_amount': predicted_amount_str}, {'predicted_date': predicted_date_str}
                            , {'predicted_sig': predicted_signature_str})


    #return render(request, 'advanced_form_components.html')
    return render(request, 'advanced_form_components.html')

def form_validation(request):
    return render(request, 'form_validation.html')

def extractData(request):
    return render(request, 'form_validation.html')

def notFound_homepage(request):
    return render(request, 'index.html')




# Create your views here.
