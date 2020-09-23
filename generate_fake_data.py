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

model = load_model('../generator_model_075_077.h5')
X = model.predict(generate_latent_points(3,1000))
rows,columns = X.shape
X_rescaled = np.zeros(X.shape)

minmaxes = loadtxt('diabetes_minmaxes.csv', delimiter=',')

print('Min/max scaling parameters for data:')
print(minmaxes)

for j in range(columns):
    min_max_difference = minmaxes[j,1] - minmaxes[j,0]
    print('min/max/diff of ', j, ': ', minmaxes[j,1], ', ', minmaxes[j,0], ', ', min_max_difference )
    for i in range(rows):
        original_value = X[i,j]
        new_value = (original_value * min_max_difference) + minmaxes[j,0]
        X_rescaled[i,j] = round(new_value)
        if j == 5:
            X_rescaled[i,j] = round(new_value, 1)
        if j == 6:
            X_rescaled[i,j] = round(new_value, 3)
        if j == columns - 1:
            X_rescaled[i,j] = round(original_value)
        if(original_value == 0):
            X_rescaled[i,j] = original_value


#print(X[0:3])
np.set_printoptions(suppress=True)
print(X)
print('First 10 columns of generated data: ', X_rescaled[:10])

np.savetxt('generated_data.csv', X_rescaled, fmt='%i,%i,%i,%i,%i,%1.1f,%1.3f,%i,%i', delimiter=',', newline='\n')
