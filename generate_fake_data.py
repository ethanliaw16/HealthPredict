from keras.models import load_model
from numpy.random import randn, rand

def generate_latent_points(latent_dimension, num_samples):
    x_input = randn(latent_dimension * num_samples)
    x_input = x_input.reshape(num_samples, latent_dimension)
    return x_input

def print_generated_data(generated, n):
    for i in range(n):
        print(generated[i])

model = load_model('../generator_model_200.h5')
X = model.predict(generate_latent_points(3,10))

print(X)
