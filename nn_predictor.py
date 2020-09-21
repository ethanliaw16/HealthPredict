from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, Dropout
from keras.optimizers import Adam
from numpy.random import randint, uniform, randn, rand
from sklearn.preprocessing import minmax_scale
import min_max
import numpy as np
import pandas as pd
import random

def predictor():
    model = Sequential()
    model.add(Dense(24, input_dim=15))
    model.add(LeakyReLU(alpha=.2))
    #model.add(Dropout(.4))
    model.add(Dense(12, activation='relu'))
    model.add(LeakyReLU(alpha=.2))
    model.add(Dropout(.4))
    model.add(Dense(1, activation='sigmoid'))
    
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def train_predictor(predictor, train_x, train_y, test_x, test_y, epochs=100, batch_size=60):
    predictor.fit(train_x, train_y, epochs=epochs, batch_size=batch_size,validation_data=(test_x, test_y))




data = pd.read_csv('/data/all_data_important_columns.csv')
X_dataframe = data[['YearOfBirth',
'State',
'HeightMedian',
'WeightMedian',
'BMIMedian',
'RangeBMI',
'SystolicBPMedian', 
'DiastolicBPMedian', 
'L2_HypertensionEssential', 
'L2_MixedHyperlipidemia',
'L2_ChronicRenalFailure',
'L2_Alcohol',
'L2_Hypercholesterolemia',
'L2_AtherosclerosisCoronary',
'L2_HyperlipOther',]]


