import numpy as np
import pandas as pd
from math import sqrt
def distance(row1,row2):
    distance = 0.0
    #print('row 1 length ', len(row1))
    #print('row 2 length ', len(row2))
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)

def k_nearest(data, input, k):
    distances = []
    neighbors = []
    data_as_lists = data.values.tolist()
    for row in data_as_lists:
        distance_from_input = distance(np.array(row),input)
        distances.append((row,distance_from_input))
    distances.sort(key=lambda tup: tup[1])
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors

def impute_missing(input_vector):
    #load gan generated data
    data = pd.read_csv('../data/ehr_generated_data_with_smoking.csv')
    input_columns = data[['YearOfBirth', 
    'Gender_M',
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
    'Smoking_2', 
    'Smoking_15', 
    'Smoking_20', 
    'Smoking_30', 
    'Smoking_40']]

    input_columns = input_columns.reset_index(drop=True)

    print('GAN data dimensions: ', data.shape)
    print('input vector size ', input_vector.size)
    non_missing = []
    missing = []
    for index in range(input_vector.size):
        if input_vector[index] >= 0:
            non_missing.append(index)
        else:    
            print('imputation needed on index ', index)
            missing.append(index)
    print('indices to use for knn impute: ', non_missing)
    known_indices = np.array(non_missing)
    input_columns['Class_Weight']=10.5
    data_known_columns_only = input_columns.iloc[:,non_missing]
    print('Known data columns: ', input_columns.columns[non_missing])
    print('First rows of known data: ', data_known_columns_only.head())
    print('Our input: ', input_vector[known_indices])
    knn = k_nearest(data_known_columns_only, input_vector[known_indices], 5)
    knn_as_df = pd.DataFrame(knn, columns=input_columns.columns[non_missing])
    knn_as_df = knn_as_df.merge(input_columns,on=knn_as_df.columns.tolist(), how='left')
    print('K nearest: ', knn)
    print('K nearest as dataframe: ', knn_as_df)
    mean_of_knn = knn_as_df.mean(axis=0)
    mean_ordered_columns = mean_of_knn[['YearOfBirth', 
    'Gender_M',
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
    'Smoking_2', 
    'Smoking_15', 
    'Smoking_20', 
    'Smoking_30', 
    'Smoking_40']]
    print('Mean row of knn: ', mean_ordered_columns)
    imputed_as_list = np.array(mean_ordered_columns.values.tolist())
    print('imputed from knn: ', imputed_as_list[missing])

    for missing_value_index in missing:
        input_vector[missing_value_index] = imputed_as_list[missing_value_index]
    print('Input with missing values filled in: ', input_vector)
    return input_vector