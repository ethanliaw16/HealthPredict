from flask import Flask, jsonify, request
from keras.models import load_model
from numpy.random import randn, rand, randint
from numpy import loadtxt
from json import JSONEncoder
from random import randrange
import json
import numpy as np
import pandas as pd
app = Flask(__name__)

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def generate_latent_points(latent_dimension, num_samples):
    x_input = randn(latent_dimension * num_samples)
    x_input = x_input.reshape(num_samples, latent_dimension)
    return x_input

def get_model():
    global model
    model = load_model('../generator_model_066_074.h5')

print('Loading generator...')
get_model()


def generate_fake_data(num_samples_to_generate):
    minmaxes = loadtxt('../diabetes_minmaxes.csv', delimiter=',')
    generated_data = model.predict(generate_latent_points(3,num_samples_to_generate))
    rows,columns = generated_data.shape
    data_rescaled = np.zeros(generated_data.shape)

    for j in range(columns):
        min_max_difference = minmaxes[j,1] - minmaxes[j,0]
        print('min/max/diff of ', j, ': ', minmaxes[j,1], ', ', minmaxes[j,0], ', ', min_max_difference )
        for i in range(rows):
            original_value = generated_data[i,j]
            new_value = (original_value * min_max_difference) + minmaxes[j,0]
            data_rescaled[i,j] = round(new_value)
            if j == 5:
                data_rescaled[i,j] = round(new_value, 1)
            if j == 6:
                data_rescaled[i,j] = round(new_value, 3)
            if j == columns - 1:
                data_rescaled[i,j] = round(original_value)
            if(original_value == 0):
                data_rescaled[i,j] = original_value
    
    return data_rescaled
    
@app.route('/home', methods=['GET'])
def get_data_pair():
    print('Recieved GET request')
    reals = loadtxt('../diabetes.csv', delimiter=',')
    fakes = generate_fake_data(768)
    random_real_point = reals[randrange(768)]
    random_fake_point = fakes[randrange(768)]
    real_dict = {'real':random_real_point}
    fake_dict = {'fake':random_fake_point}
    encodedReals = json.dumps(random_real_point, cls=NumpyArrayEncoder)
    encodedFakes = json.dumps(random_fake_point, cls=NumpyArrayEncoder)
    response = {'real':encodedReals, 'fake':encodedFakes}
    return jsonify(response)


