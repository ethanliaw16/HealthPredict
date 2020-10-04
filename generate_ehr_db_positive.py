from keras.models import load_model
from numpy.random import randn, rand
from numpy import loadtxt
import numpy as np
import pandas as pd 

def get_data_min_max(dataset):
    _,num_columns = dataset.shape
    data_minmaxes = []
    for (columnName, columnData) in dataset.iteritems():
        column_min = columnData.min()
        column_max = columnData.max()
        #print('Column ', i, ' min: ', column_min, ' max: ', column_max)
        data_minmaxes.append([column_min,column_max])
    return data_minmaxes

def map_gender_column(row):
    if(row['Gender_M'] == 1):
        return 'M'
    return 'F'

def map_female_column(row):
    if(row['Gender_M'] == 1):
        return 0
    return 1

def generate_latent_points(latent_dimension, num_samples):
    x_input = randn(latent_dimension * num_samples)
    x_input = x_input.reshape(num_samples, latent_dimension)
    return x_input

def print_generated_data(generated, n):
    for i in range(n):
        print(generated[i])

model = load_model('./trained_models/ehr_generator_model_with_smoking_089_085.h5')
X = model.predict(generate_latent_points(15,1500))
rows,columns = X.shape
X_rescaled = np.zeros(X.shape)
minmaxes = loadtxt('./data/ehr_minmaxes_with_smoking.csv', delimiter=',')

print('Min/max scaling parameters for data:')
print(minmaxes)

for j in range(columns - 1):
    min_max_difference = minmaxes[j,1] - minmaxes[j,0]
    print('min/max/diff of ', j, ': ', minmaxes[j,1], ', ', minmaxes[j,0], ', ', min_max_difference )
    for i in range(rows):
        original_value = X[i,j]
        new_value = (original_value * min_max_difference) + minmaxes[j,0]
        X_rescaled[i,j] = round(new_value, 5)
        if j == 0:
            X_rescaled[i,j] = round(new_value)
        if j > 5:
                X_rescaled[i,j] = round(new_value)
X_rescaled[:,20] = 1
#print(X[0:3])
np.set_printoptions(suppress=True)
print(X)
#outcomes = X_rescaled[1:10,20];
print('First 10 columns of generated data: ', X_rescaled[:10])
X_rescaled_df = pd.DataFrame(data=X_rescaled, columns=['YearOfBirth', 
'HeightMedian', 
'WeightMedian', 
'BMIMedian', 
'SystolicBPMedian', 
'DiastolicBPMedian',
'Gender_M',
'L2_HypertensionEssential', 
'L2_MixedHyperlipidemia', 
'L2_ChronicRenalFailure', 
'L2_Alcohol', 
'L2_Hypercholesterolemia', 
'L2_AtherosclerosisCoronary',
'L2_HyperlipOther',
'Smoking_0',
'Smoking_2',
'Smoking_15',
'Smoking_20',
'Smoking_30',
'Smoking_40',
'DMIndicator'])
X_rescaled_df['Gender'] = X_rescaled_df.apply(lambda row: map_gender_column(row), axis = 1)
X_rescaled_df['Gender_F'] = X_rescaled_df.apply(lambda row: map_female_column(row), axis = 1)
females_db = X_rescaled_df.loc[X_rescaled_df['Gender']=='F']
males_db = X_rescaled_df.loc[X_rescaled_df['Gender']=='M']
print('First 10 columns of female datapoints: ', females_db[:10])
print('female mean height ', females_db['HeightMedian'].mean())
print('female mean weight ', females_db['WeightMedian'].mean())
print('male mean height ', males_db['HeightMedian'].mean())
print('male mean weight ', males_db['WeightMedian'].mean())
print('smoke 0 values', X_rescaled_df['Smoking_0'].value_counts())
print('smoke 2 values', X_rescaled_df['Smoking_2'].value_counts())
print('smoke 15 values', X_rescaled_df['Smoking_20'].value_counts())
print('smoke 20 values', X_rescaled_df['Smoking_15'].value_counts())
print('smoke 30 values', X_rescaled_df['Smoking_30'].value_counts())
print('smoke 40 values', X_rescaled_df['Smoking_40'].value_counts())
#print(X_rescaled_df['YearOfBirth'].value_counts())
print(X_rescaled_df['Gender'].value_counts())
X_rescaled_df.to_csv('./data/ehr_generated_data_with_smoking.csv', index=False)
