from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten, Dropout 
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import AveragePooling2D
from PIL import Image

import numpy as np
from numpy import linalg as LA
import os

dir1 = ['static/image/training/acral melanoma/',
        'static/image/testing/acral melanoma/',
        'static/image/training/benign nevi/',
        'static/image/testing/benign nevi/']

def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]

def preprocess(img,input_size):
    nimg = img.convert('RGB').resize(input_size)
    img_arr = (np.array(nimg))
    return img_arr

def reshape(imgs_arr):
    return np.stack(imgs_arr, axis=0)

class FeatureExtractor:
    def __init__(self):
        inputs = Input(shape=(224,224,3))

        inputs = Input(shape=(224,224,3))
        
        conv1 = Conv2D(32, kernel_size=3, strides = 1, activation='relu')(inputs)
        pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)
        
        conv1 = Conv2D(64, kernel_size=3, strides = 1, activation='relu')(pool1)
        pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)

        conv1 = Conv2D(128, kernel_size=3, strides = 1, activation='relu')(pool1)
        pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)

        conv1 = Conv2D(256, kernel_size=3, strides = 1, activation='relu')(pool1)
        pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)

        conv1 = Conv2D(512, kernel_size=3, strides = 1, activation='relu')(pool1)
        pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)

        flat = Flatten()(pool1)
        drop_out1 = Dropout(0.5)(flat)
        hidden1 = Dense(2048,activation='relu', name='fc1')(drop_out1)
        hidden2 = Dense(500,activation='relu', name='fc3')(hidden1)

        self.model3 = Model(inputs=inputs, outputs=hidden2)

    def extract_features(self, img_path):
        im = Image.open(img_path)
        X = preprocess(im,(224,224))
        X = reshape([X])

        feat = self.model3.predict(X)
        norm_feat = feat[0]/LA.norm(feat[0])
        return norm_feat

model3 = FeatureExtractor()

def ekstrak_train():
    feats = []
    names = []
    dir_training = [dir1[0], dir1[2]]

    for direktori in dir_training:
        img_list = get_imlist(direktori)
        for i, img_path in enumerate(img_list):
            norm_feat = model3.extract_features(img_path)
            img_name = os.path.split(img_path)[1]
            feats.append(norm_feat)
            if 'AM' in img_name:
                names.append('acral melanoma')
            else:
                names.append('benign nevi')
            print("extracting feature from image")

    feats = np.array(feats)
    names = np.array(names)

    return feats, names

def ekstrak_test(img_path):
    feats_test = []
    names_test = []
    norm_feat = model3.extract_features(img_path)
    img_name = os.path.split(img_path)[1]
    feats_test.append(norm_feat)
    if 'AM' in img_name:
        names_test.append('acral melanoma')
    else:
        names_test.append('benign nevi')
    print("extracting feature from image")


    feats_test = np.array(feats_test)
    names_test = np.array(names_test)

    return feats_test, names_test
