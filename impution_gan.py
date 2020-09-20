from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, Dropout
from keras.optimizers import Adam
from numpy.random import randint, uniform, randn, rand
from sklearn.preprocessing import minmax_scale
import numpy as np
import pandas as pd
import random

dataset = pd.read_csv('ehr_diabetes.csv', delimiter=',')
print(dataset.shape)

print(dataset["race"].head())
dataset_weight_known = dataset[dataset["weight"] != '?']
print(dataset_weight_known.shape)
print(dataset_weight_known.head())

columns_with_many_missing_values = []

for(column_name, column_data) in dataset.iteritems():
    num_missing = (dataset[column_name] == '?').sum()
    if num_missing > 100:
        columns_with_many_missing_values.append((column_name, num_missing))

print(columns_with_many_missing_values)

dataset_less_columns = dataset.drop(columns=['payer_code','medical_specialty', 'encounter_id', 'patient_nbr'])
print('Shape of dataset with dropped columns: ', dataset_less_columns.shape)
dataset_no_missing = dataset_less_columns[(dataset_less_columns != '?').all(axis=1)]
print('Shape of data without missing values: ', dataset_no_missing.shape)
print('Diagnosis 1: ', (dataset_no_missing['diag_1'].str.match('250')).sum())
print('Diagnosis 2: ', (dataset_no_missing['diag_2'].str.match('250')).sum())
print('Diagnosis 3: ', (dataset_no_missing['diag_3'].str.match('250')).sum())
print(dataset_no_missing[dataset_no_missing['diag_3'].str.match('250')]['diag_3'])
#print(dataset_no_missing['diag_2'].head(10))
#print(dataset_no_missing['diag_1'].head())
#print(dataset_no_missing['diag_1'].value_counts())

dataset_no_missing.to_csv('ehr_diabetes_no_missing_3k.csv')