from numpy import loadtxt
import numpy as np
import sys

def get_data_min_max(dataset):
    _,num_columns = dataset.shape
    data_minmaxes = []
    for (columnName, columnData) in dataset.iteritems():
        column_min = columnData.min()
        column_max = columnData.max()
        #print('Column ', i, ' min: ', column_min, ' max: ', column_max)
        data_minmaxes.append([column_min,column_max])
    return data_minmaxes

filename = sys.argv[1]
dataset = loadtxt(filename, delimiter=',')

file_without_extension = filename[:filename.index('.csv')]

print('file without extension ', file_without_extension)
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

minmaxes_array = np.asarray(data_minmaxes)
print(data_minmaxes)

new_filename = file_without_extension + '_minmaxes.csv'
print('Mins and maxes saved to ', new_filename)
np.savetxt(new_filename,minmaxes_array, delimiter=',')
