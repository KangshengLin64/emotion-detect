import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
from joblib import dump
from sklearn.preprocessing import MinMaxScaler



def sauvola_threshold(image, window_size=5, k=0.15, r=128):
    

    # Compute the local mean and standard deviation of the image
    mean = cv2.blur(image, (window_size, window_size))
    std = np.sqrt(cv2.blur(np.square(image), (window_size, window_size)) - np.square(mean))

    # Compute the threshold using Sauvola's algorithm
    threshold = mean * (1 + k * ((std / r) - 1))

    # Binarize the image using the computed threshold
    binary = np.zeros_like(image)
    binary[image > threshold] = 255

    return binary

#img process
def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    #img = cv2.resize(img, (100, 100))
    img=cv2.equalizeHist(img)
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.addWeighted(img, 1.7, blurred, -0.5, 0)
    
    img= sauvola_threshold(img, window_size=5, k=0.15, r=128)
    
    imgm= (img*0.008).astype('uint8')
    img=cv2.multiply(imgm, img)
    
 
    return img

#HOG fearture
def extract_hog_features(image):
    # Extract HOG features from the image
    features, _ = hog(image, orientations=6, pixels_per_cell=(6, 6), cells_per_block=(3, 3), visualize=True, transform_sqrt=True)

    return features


#LBP feature 
def extract_LBP_features(image):
    radius = 4
    n_points = 8 * radius

    lbp = np.zeros_like(image)
    for i in range(radius, image.shape[0]-radius):
        for j in range(radius, image.shape[1]-radius):
            center = image[i, j]
            values = []
            for k in range(n_points):
                x = i + int(radius * np.cos(2 * np.pi * k / n_points))
                y = j - int(radius * np.sin(2 * np.pi * k / n_points))
                values.append(image[x, y])
            values = np.array(values)
            lbp_value = np.sum((values >= center) * (2 ** np.arange(n_points)))
            lbp[i, j] = lbp_value
    
    # Reshape LBP to 2D array
    
    return lbp



#label database
def label_database(directory):
    labels = []
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    for c in emotions:
        class_dir = os.path.join(directory, c)
        for image_file in os.listdir(class_dir):
            labels.append(c)
    return np.array(labels)


#image database

def image_database(directory):
    images=[]
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    for i in emotions:
        emotion_dir= os.path.join(directory,i)
        for imgfile in os.listdir(emotion_dir):
            image_path = os.path.join(emotion_dir, imgfile)
            image=read_image(image_path)
            images.append(image)
    return np.array(images)


    
#load database
print('start load')
trainx=image_database('C:/Users/70937/Desktop/csi/csc475/dataset/train')
trainy=label_database('C:/Users/70937/Desktop/csi/csc475/dataset/train')

testx=image_database('C:/Users/70937/Desktop/csi/csc475/dataset/test')
testy=label_database('C:/Users/70937/Desktop/csi/csc475/dataset/test')
print('finish load')


#hog
print('start hog')
trainx_hog = []
for image in trainx:
    features = extract_hog_features(image)
    trainx_hog.append(features)
trainx_hog = np.array(trainx_hog)

testx_hog = []
for image in testx:
    features = extract_hog_features(image)
    testx_hog.append(features)
testx_hog = np.array(testx_hog)
print('finish hog')


#LBP
print('start lbp')
trainx_lbp = []
for image in trainx:
    features = extract_LBP_features(image)
    trainx_lbp.append(features)
trainx_lbp = np.array(trainx_lbp)
trainx_lbp =trainx_lbp.reshape(trainx_lbp.shape[0], -1)

testx_lbp = []
for image in testx:
    features = extract_LBP_features(image)
    testx_lbp.append(features)
testx_lbp = np.array(testx_lbp)
testx_lbp =testx_lbp.reshape(testx_lbp.shape[0], -1)
print('finish lbp')

print(trainx_lbp.shape,testx_lbp.shape)
features_train= np.concatenate((trainx_hog, trainx_lbp), axis=1)
features_test = np.concatenate((testx_hog, testx_lbp), axis=1)

# train the model
svm = SVC(kernel='rbf', decision_function_shape='ovr', C=1.0,gamma=0.1,tol=1e-5)

print('start to train')
svm.fit(trainx_hog, trainy)
print('finish train')

y_pred = svm.predict(testx_hog)
print('Accuracy:', accuracy_score(testy, y_pred))
print('Precision:', precision_score(testy, y_pred, average='weighted', zero_division=1))
print('Recall:', recall_score(testy, y_pred, average='weighted'))
#save model
print('start save mod')
dump(svm, 'emotion_detect.joblib')  