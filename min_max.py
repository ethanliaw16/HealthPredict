import numpy as np

def get_data_min_max(dataset):
    _,num_columns = dataset.shape
    data_minmaxes = []
    for i in range(num_columns):
        column = dataset[:,i]
        column_min = column.min()
        column_max = column.max()
        if i != 0: 
            column = column[np.nonzero(column)]
            column_min = column.min()
        #print('Column ', i, ' min: ', column_min, ' max: ', column_max)
        data_minmaxes.append([column_min,column_max])
    return data_minmaxes