import cv2
import joblib
from skimage.feature import hog
import numpy as np

# Load the SVM model
svm_model = joblib.load('emotion_detect.joblib')

# Load the new image
img = cv2.imread('C:/Users/70937/Desktop/csi/csc475/14.jpg')

# Preprocess the image
img = cv2.resize(img, (48, 48))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Extract features from the preprocessed image
# Replace this with your own feature extraction method
features = hog(gray, orientations=6, pixels_per_cell=(6,6), cells_per_block=(3,3), visualize=False, transform_sqrt=True)
features = features.reshape(1, -1)
# Use the predict method of the SVM model to predict the emotion label
emotion_label = svm_model.predict(features)

print('The predicted emotion label for the new image is:', emotion_label)