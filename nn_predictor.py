from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, Dropout
from keras.optimizers import Adam
from numpy.random import randint, uniform, randn, rand
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

import min_max
import numpy as np
import pandas as pd
import tensorflow as tf
import random

def predictor():
    model = Sequential()
    model.add(Dense(24, input_dim=20))
    model.add(LeakyReLU(alpha=.2))
    #model.add(Dropout(.4))
    model.add(Dense(12, activation='relu'))
    model.add(LeakyReLU(alpha=.2))
    model.add(Dropout(.4))
    initial_bias = np.log([1903/8044])
    output_bias = tf.keras.initializers.Constant(initial_bias)
    print(output_bias)
    model.add(Dense(1, activation='sigmoid'))
    
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def train_predictor(predictor, train_x, train_y, test_x, test_y, epochs=100, batch_size=2048):
    class_weight = {0: 1.,1: 100.}
    predictor.fit(train_x, train_y, epochs=epochs, batch_size=batch_size,validation_data=(test_x, test_y), class_weight=class_weight)

data = pd.read_csv('./data/all_data_important_columns.csv')
data.rename(columns={'Smoker.x': 'Smoke'},inplace = True)
print(data['DMIndicator'].value_counts())
#print(data.head())
encoded_gender = pd.get_dummies(data.Gender, prefix='Gender')
encoded_smoking = pd.get_dummies(data.Smoke, prefix='Smoke')
encoded_gender.reset_index(drop=True)
encoded_smoking.reset_index(drop=True)
data.reset_index(drop=True)
print(encoded_gender.head())
data = pd.concat([data, encoded_gender], axis=1)
data = pd.concat([data,encoded_smoking], axis=1)
data['L2_HypertensionEssential'] = data['L2_HypertensionEssential'].clip(0,1)
data['L2_MixedHyperlipidemia'] = data['L2_MixedHyperlipidemia'].clip(0,1)
data['L2_ChronicRenalFailure'] = data['L2_ChronicRenalFailure'].clip(0,1)
data['L2_Alcohol'] = data['L2_Alcohol'].clip(0,1)
data['L2_Hypercholesterolemia'] = data['L2_Hypercholesterolemia'].clip(0,1)
data['L2_AtherosclerosisCoronary'] = data['L2_AtherosclerosisCoronary'].clip(0,1)
data['L2_HyperlipOther'] = data['L2_HyperlipOther'].clip(0,1)
actual_diabetics = data.loc[data['DMIndicator']==1]
actual_nondiabetics = data.loc[data['DMIndicator']==0]

actual_diabetics.to_csv('./data/minority_ehr_encoded.csv', index=False)

X_dataframe = data[[
'Gender_M', 
'Gender_F', 
'YearOfBirth', 
'HeightMedian', 
'WeightMedian', 
'BMIMedian', 
'SystolicBPMedian', 
'DiastolicBPMedian', 
'L2_HypertensionEssential', 
'L2_MixedHyperlipidemia', 
'L2_ChronicRenalFailure', 
'L2_Alcohol', 
'L2_Hypercholesterolemia', 
'L2_AtherosclerosisCoronary',
'L2_HyperlipOther',
'Smoke_2', 
'Smoke_15', 
'Smoke_20', 
'Smoke_30', 
'Smoke_40']]

X_diabetic = actual_diabetics[[
'Gender_M', 
'Gender_F', 
'YearOfBirth', 
'HeightMedian', 
'WeightMedian', 
'BMIMedian', 
'SystolicBPMedian', 
'DiastolicBPMedian', 
'L2_HypertensionEssential', 
'L2_MixedHyperlipidemia', 
'L2_ChronicRenalFailure', 
'L2_Alcohol', 
'L2_Hypercholesterolemia', 
'L2_AtherosclerosisCoronary',
'L2_HyperlipOther',
'Smoke_2', 
'Smoke_15', 
'Smoke_20', 
'Smoke_30', 
'Smoke_40']]

X_nondiabetic = actual_nondiabetics[[
'Gender_M', 
'Gender_F', 
'YearOfBirth', 
'HeightMedian', 
'WeightMedian', 
'BMIMedian', 
'SystolicBPMedian', 
'DiastolicBPMedian', 
'L2_HypertensionEssential', 
'L2_MixedHyperlipidemia', 
'L2_ChronicRenalFailure', 
'L2_Alcohol', 
'L2_Hypercholesterolemia', 
'L2_AtherosclerosisCoronary',
'L2_HyperlipOther',
'Smoke_2', 
'Smoke_15', 
'Smoke_20', 
'Smoke_30', 
'Smoke_40']]


y = data[['DMIndicator']]
print(y.value_counts())
y_diabetic = actual_diabetics['DMIndicator']
y_nondiabetic = actual_nondiabetics['DMIndicator']
X_nondiabetic_small = X_nondiabetic[:1800]
y_nondiabetic_small = y_nondiabetic[:1800]

X_balanced = pd.concat([X_nondiabetic_small, X_diabetic])
y_balanced = pd.concat([y_nondiabetic_small, y_diabetic])
print('undersampled nondiabetic shape', X_balanced.shape)
X_scaled = minmax_scale(X_balanced)
print(X_scaled[:5])
train_X,test_X,train_y,test_y = train_test_split(X_scaled, y_balanced, random_state=42)
#train_X = X_scaled[:7000]
#train_y = y[:7000]
#
#test_X = X_scaled[7000:]
#test_y = y[7000:]

print(train_X.shape)
print(test_X.shape)

model = predictor()
#model = make_model(output_bias=.19)
print(model.summary())

train_predictor(model, train_X,train_y, test_X,test_y)

results = model.evaluate(test_X, test_y, batch_size=128)

print(results)

Y_pred = model.predict(test_X)
y_pred = np.argmax(Y_pred, axis=1)

print(y_pred[:10,])
print('Confusion Matrix\n',confusion_matrix(test_y,y_pred))

print('diabetic shape ', X_diabetic.shape)
print('nondiabetic shape ', X_nondiabetic.shape)
all_diabetic_pred = model.predict(minmax_scale(X_diabetic))
diabetic_pred = np.argmax(all_diabetic_pred, axis=1)

print('Confusion Matrix for All Diabetics\n', confusion_matrix(y_diabetic,diabetic_pred))

all_nondiabetic_pred = model.predict(minmax_scale(X_nondiabetic))
nondiabetic_pred = np.argmax(all_nondiabetic_pred, axis=1)

print('Confusion Matrix for All Nondiabetics\n', confusion_matrix(y_nondiabetic, nondiabetic_pred))

print('Diabetics accuracy', np.mean(diabetic_pred))
print('Nondiabetics accuracy', 1 - np.mean(nondiabetic_pred))
