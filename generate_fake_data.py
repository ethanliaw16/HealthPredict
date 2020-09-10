from keras.models import load_model
from numpy.random import randn, rand
from numpy import loadtxt
import numpy as np

def generate_latent_points(latent_dimension, num_samples):
    x_input = randn(latent_dimension * num_samples)
    x_input = x_input.reshape(num_samples, latent_dimension)
    return x_input

def print_generated_data(generated, n):
    for i in range(n):
        print(generated[i])

model = load_model('../generator_model_066_070.h5')
X = model.predict(generate_latent_points(3,10))
rows,columns = X.shape
X_rescaled = np.zeros(X.shape)
print('Empty array for rescaled data: ', X_rescaled)
minmaxes = loadtxt('diabetes_minmaxes.csv', delimiter=',')
print(minmaxes)

for j in range(columns):
    min_max_difference = minmaxes[j,1] - minmaxes[j,0]
    print('min/max/diff of ', j, ': ', minmaxes[j,1], ', ', minmaxes[j,0], ', ', min_max_difference )
    for i in range(rows):
        original_value = X[i,j]
        new_value = (original_value * min_max_difference) + minmaxes[j,0]
        X_rescaled[i,j] = new_value
        if j == 0:
            X_rescaled[i,j] = round(new_value)
        if j == columns - 1:
            X_rescaled[i,j] = round(original_value)
        
#print(X[0:3])
print(X_rescaled)
