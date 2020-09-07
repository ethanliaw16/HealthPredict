from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from numpy.random import randint, uniform
import numpy as np

# load the dataset
dataset = loadtxt('diabetes.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]

print('Train', X.shape, y.shape)
print('Test', X.shape, y.shape)

def discriminator():
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def generate_real_samples(dataset, num_samples):
    random_index = randint(0,dataset.shape[0], num_samples)
    X = dataset[random_index]
    reals = np.ones((num_samples, 1))
    return X, reals

def generate_fake_samples(num_samples, insulin_feature_present):
    pregnancies = randint(0,10, size=(num_samples,1))
    glucose = randint(70,200, size=(num_samples,1))
    d_blood_pressure = randint(50,150,size=(num_samples,1))
    tri_skin_fold_thickness = randint(0,100, size=(num_samples,1))
    insulin = np.zeros((num_samples,1))
    if(insulin_feature_present):
        insulin = randint(0,800, size=(num_samples,1))
    bmi = uniform(low=16.0, high=35.0,size=(num_samples,1))
    diabetes_pedigree = uniform(low=0,high=.900,size=(num_samples,1))
    age = randint(20,85,size=(num_samples,1))
    outcome = np.zeros((num_samples,1))
    return np.column_stack((pregnancies,glucose,d_blood_pressure,tri_skin_fold_thickness,insulin,bmi,diabetes_pedigree,age)),outcome


discriminator_model = discriminator()
discriminator_model.summary()

fakes = generate_fake_samples(10,False)
reals = generate_real_samples(X,10)
print(reals)
print(fakes)


