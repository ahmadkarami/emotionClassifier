import os
import numpy as asa
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from keras.preprocessing.image import load_img
from tensorflow import keras
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential, Model
from keras.layers import concatenate
from keras.callbacks import *
from tensorflow.keras.utils import to_categorical
from keras.layers import Dense, BatchNormalization, Conv2D, MaxPooling2D, Dropout, Activation
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.initializers import RandomNormal, Constant
from data_preparation import *


def stack(availale_models):
    lable = []
    predicted = []
    
    key_list = list(EMOTIONS.keys())
    val_list = list(EMOTIONS.values())
    
    for model in availale_models:
        for subdir, dirs, files in os.walk("test"):
            x = len(files)
            if x != 0:
                for file in files:
                    
                    img = load_img(subdir + '/' + file, target_size=(150, 150))
                    img = np.expand_dims(img, axis=0)
                    
                    result = model.predict(img)
                    predicted.append(result[0])
                    
                    index = key_list[val_list.index(subdir.split('\\')[1])]
                    lable.append(index)
                    
    
    X = np.array(predicted)
    y = np.array(lable)
    
    return X,y


def load_all_models():
    
    all_models = list()
    cnn = keras.models.load_model("emotionClassifierModel_0.hdf5")     
    vgg = keras.models.load_model("emotionClassifierModel_1.hdf5")
    
    all_models.append(cnn)
    all_models.append(vgg)
    	
    return all_models

# fit a model based on the outputs from the ensemble members
def fit_model(inputX, inputy):
	model = LogisticRegression()
	model.fit(inputX, inputy)
	return model

# make a prediction with the stacked model
def stacked_prediction(model, inputX):
	yhat = model.predict(inputX)
	return yhat


members = load_all_models()
X, y = stack(members)
n_train = 30
trainX, testX = X[:n_train,:], X[n_train:,:]
trainy, testy = y[:n_train], y[n_train:]

model = fit_model(testX, testy)
yhat = stacked_prediction(model, testX)
acc = accuracy_score(testy, yhat)
print('Stacked Test Accuracy: %.3f' % acc)
