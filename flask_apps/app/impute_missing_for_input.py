import numpy as np
import pandas as pd
def impute_missing(input_vector):
    #load gan generated data
    data = pd.read_csv('../data/all_data_important_columns.csv')
    print('input vector size ', input_vector.size)
    for index in range(input_vector.size):
        if input_vector[index] < 0:
            print('imputation needed on index ', index)
    return input_vector