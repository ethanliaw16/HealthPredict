from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, Dropout
from keras.optimizers import Adam
from numpy.random import randint, uniform, randn, rand
from sklearn.preprocessing import minmax_scale
from get_mins_maxes_from_data import get_data_min_max
import numpy as np
import pandas as pd
import random

#GAN configured for EHR data. This model aims to generate datapoints to mimic those of 
#the Fusion EHR dataset from kaggle. 

def discriminator():
    model = Sequential()
    model.add(Dense(512, input_dim=15))
    model.add(LeakyReLU(alpha=.2))
    model.add(Dropout(.4))
    model.add(Dense(256, activation='relu'))
    model.add(LeakyReLU(alpha=.2))
    model.add(Dropout(.4))
    model.add(Dense(1, activation='sigmoid'))
    
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def generator(latent_dimension):
    model = Sequential()
    model.add(Dense(512, input_dim=latent_dimension))
    model.add(LeakyReLU(alpha=.2))
    model.add(Dropout(.4))
    model.add(Dense(256, activation='relu'))
    model.add(LeakyReLU(alpha=.2))
    model.add(Dropout(.4))
    model.add(Dense(15, activation = 'relu'))
    opt = Adam(lr=.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def generate_real_samples(dataset, num_samples):
    random_index = randint(0,dataset.shape[0], num_samples)
    X = dataset[random_index]
    reals = np.ones((num_samples, 1))
    return X, reals

def generate_fake_samples(num_samples):
    height = rand(num_samples,1)
    weight = rand(num_samples,1)
    s_blood_pressure = rand(num_samples,1)
    d_blood_pressure = rand(num_samples,1)
    bmi = rand(num_samples,1)
    birth_year = rand(num_samples,1)
    possible_outcomes = [0,1]
    gender_M = []
    L2_HypertensionEssential = []
    L2_MixedHyperlipidemia = []
    L2_ChronicRenalFailure = []
    L2_Alcohol = []
    L2_Hypercholesterolemia = []
    L2_AtherosclerosisCoronary = []
    L2_HyperlipOther = []
    #Smoke_0 = []
    #Smoke_2 = []
    #Smoke_15 = []
    #Smoke_20 = []
    #Smoke_30 = []
    #Smoke_40 = []
    for i in range(num_samples):
        gender_M.append(random.choice(possible_outcomes))
        L2_HypertensionEssential.append(random.choice(possible_outcomes))
        L2_MixedHyperlipidemia.append(random.choice(possible_outcomes))
        L2_ChronicRenalFailure.append(random.choice(possible_outcomes))
        L2_Alcohol.append(random.choice(possible_outcomes))
        L2_Hypercholesterolemia.append(random.choice(possible_outcomes))
        L2_AtherosclerosisCoronary.append(random.choice(possible_outcomes))
        L2_HyperlipOther.append(random.choice(possible_outcomes))
    dm_indicator = np.ones((num_samples, 1))
    fake_labels = np.zeros((num_samples,1))
    return np.column_stack((height,weight,s_blood_pressure,d_blood_pressure,bmi,birth_year, gender_M, L2_HypertensionEssential, L2_MixedHyperlipidemia, L2_ChronicRenalFailure, L2_Alcohol, L2_Hypercholesterolemia, L2_AtherosclerosisCoronary, L2_HyperlipOther, dm_indicator)),fake_labels

def generate_fake_samples_with_model(generator_model, latent_dimension, num_samples):
    x_input = generate_latent_points(latent_dimension, num_samples)
    X = generator_model.predict(x_input)
    y = np.zeros((num_samples, 1))
    return X, y

def generate_latent_points(latent_dimension, num_samples):
    x_input = randn(latent_dimension * num_samples)
    x_input = x_input.reshape(num_samples, latent_dimension)
    return x_input

def train_discriminator(model, dataset, num_iterations=150, num_batches=128):
    half_batch = int(num_batches/2)
    for i in range (num_iterations):
        # get randomly selected 'real' samples
        X_real, y_real = generate_real_samples(dataset, half_batch)
        # update discriminator on real samples
        _, real_acc = model.train_on_batch(X_real, y_real)
		# generate 'fake' examples
        X_fake, y_fake = generate_fake_samples(half_batch)
		# update discriminator on fake samples
        _, fake_acc = model.train_on_batch(X_fake, y_fake)
		# summarize performance
        print('>%d real=%.0f%% fake=%.0f%%' % (i+1, real_acc*100, fake_acc*100))

def build_gan(generator_model, discriminator_model):
    discriminator_model.trainable = False
    gan_model = Sequential()
    gan_model.add(generator_model)
    gan_model.add(discriminator_model)
    opt = Adam(lr=0.0002, beta_1=0.5)
    gan_model.compile(loss='binary_crossentropy', optimizer=opt)
    return gan_model

def performance_summary(epoch, generator_model, discriminator_model, dataset, latent_dimension, num_samples = 100):
    X_real, y_real = generate_real_samples(dataset, num_samples)
    _, acc_real = discriminator_model.evaluate(X_real, y_real, verbose = 0)
    x_fake, y_fake = generate_fake_samples_with_model(generator_model, latent_dimension, num_samples)
    _, acc_fake = discriminator_model.evaluate(x_fake, y_fake, verbose = 0)
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
    return (acc_real*100, acc_fake*100)

def train_complete_model(generator_model, 
discriminator_model, 
gan_model, 
dataset, 
latent_dimension, 
num_epochs=100, 
batch_size=32):
    batches_per_epoch = int(dataset.shape[0] / batch_size)
    half_batch = int(batch_size / 2)
    converged = False
    for i in range(num_epochs):
        for j in range(batches_per_epoch):
            X_real, y_real = generate_real_samples(scaled_dataset, half_batch)
            X_fake, y_fake = generate_fake_samples_with_model(generator_model, latent_dimension, half_batch)
            X, y = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))
            discriminator_loss, _ = discriminator_model.train_on_batch(X, y)
            X_gan = generate_latent_points(latent_dimension, batch_size)
            y_gan = np.ones((batch_size, 1))
            generator_loss = gan_model.train_on_batch(X_gan, y_gan)
            if (i + 1) % 10 == 0:
                print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, batches_per_epoch, discriminator_loss, generator_loss))
            if(i + 1) % 100 == 0:
                print('Summary----------------------------------------')
                real_acc, fake_acc = performance_summary(i, generator_model, discriminator_model, scaled_dataset, latent_dimension)
                if real_acc > 70 and fake_acc > 70 and fake_acc < 90:
                    filename = './trained_models/ehr_generator_model_%03d_%03d.h5' % (real_acc, fake_acc)
                    print('saving ', filename)
                    generator_model.save(filename)
                

# load the dataset
dataset = pd.read_csv('./data/minority_ehr_encoded.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[[ 
'YearOfBirth', 
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
'L2_HyperlipOther']]
y = dataset['DMIndicator']
#y = np.reshape(y,(np.size(y), 1))
data_minmaxes = np.asarray(get_data_min_max(X))
np.set_printoptions(suppress=True)
np.savetxt('./data/ehr_minmaxes.csv', data_minmaxes, delimiter=',', fmt='%f')

scaled_dataset = np.column_stack((minmax_scale(X),y))
print('First 5 scaled rows: ', scaled_dataset[:5])

print('Scaled', scaled_dataset.shape)

discriminator_model = discriminator()

train_discriminator(discriminator_model,scaled_dataset)

latent_dimension = 15
generator_model = generator(latent_dimension)

gan_model = build_gan(generator_model, discriminator_model)

gan_model.summary()

mins_and_maxes = get_data_min_max(X)

train_complete_model(generator_model, discriminator_model, gan_model, scaled_dataset, latent_dimension)
print('10 fakes from model', generate_fake_samples_with_model(generator_model,15,10))
#train_discriminator(discriminator_model, dataset)
fakes = generate_fake_samples(10)
reals = generate_real_samples(minmax_scale(X),10)
print('first 4 reals', scaled_dataset[:4])



