# Automatic bank checks data extractor
An End to End Computer vision project Developed for the pre-processing of IDRBT Bank checks Dataset to automatically detect and extract Handwritten text, Handwritten amount of money, and handwritten date along with signature verification of the check owner.

# Quick facts about the models used for prediction:
-Handwritten digits model(CNN) for the prediction of the courtesy amount and the Date of the check is following a VGG16-like architecture.  
-HTR system for the English Handwritten words recognition using the IAM dataset by **Dr. Harald Scheidl**.

The project is in its Beta-Version, further improvements will be available in a future update:  
# Fixes TODOs:  
-More data collection for the Arabic version and its variation ( more classes, more handwritten types)  
-Image-Preprocessing parameters automatization (bilateral filter & Haugh ligne)   
-Handwritten Arabic words model hyperparameter tuning  
-Improvement for the words segmentation   

-Link to the IDRBT dataset: https://www.idrbt.ac.in/icid.html

**Credits to Dr. Harald Scheidl** for the HTR model used within the project, you can find a link to his Github repository here:  
https://github.com/githubharald/SimpleHTR  
And a link to his medium article here:  
https://towardsdatascience.com/build-a-handwritten-text-recognition-system-using-tensorflow-2326a3487cd5  

# Keywords
Business Understanding/Analytic Approach, Data collection, Image Pre-processing, word segmentation, CNN, open-source HTR (Handwritten Text Recognition), MNIST Dataset, IAM Dataset, Custom Arabic Dataset, Deployment.
# Technologies
Python3 (3.8.12), Numpy, Matplotlib, seaborn, OpenCV, TensorFlow, Keras, Django, JavaScript, HTML, CSS.

# Developed by  
**Adem Youssef** 4th Grade data science student @ESPRIT-School of Engineering  
LinkedIn account: https://www.linkedin.com/in/adem-youssef-277019176/
