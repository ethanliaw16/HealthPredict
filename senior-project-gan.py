from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, Dropout
from keras.optimizers import Adam
from numpy.random import randint, uniform, randn, rand
from sklearn.preprocessing import minmax_scale
import numpy as np
import random

#A preliminary implementation of a GAN. This model aims to generate datapoints to mimic those of 
#the Pima Indian Diabetes dataset. This is meant to be a template for GAN that we may build in the future. 

def discriminator():
    model = Sequential()
    model.add(Dense(512, input_dim=9))
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
    model.add(Dense(256, input_dim=latent_dimension))
    model.add(Dense(9, activation = 'relu'))
    opt = Adam(lr=.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def generate_real_samples(dataset, num_samples):
    random_index = randint(0,dataset.shape[0], num_samples)
    X = dataset[random_index]
    reals = np.ones((num_samples, 1))
    return X, reals

def generate_fake_samples(num_samples, insulin_feature_present):
    pregnancies = rand(num_samples)
    glucose = rand(num_samples,1)
    d_blood_pressure = rand(num_samples,1)
    tri_skin_fold_thickness = rand(num_samples,1)
    insulin = rand(num_samples,1)
    
    bmi = rand(num_samples,1)
    diabetes_pedigree = rand(num_samples,1)
    age = rand(num_samples,1)
    possible_outcomes = [0,1]
    generated_outcomes = []
    for i in range(num_samples):
        generated_outcomes.append(random.choice(possible_outcomes))
    fake_labels = np.zeros((num_samples,1))
    return np.column_stack((pregnancies,glucose,d_blood_pressure,tri_skin_fold_thickness,insulin,bmi,diabetes_pedigree,age,generated_outcomes)),fake_labels

def generate_fake_samples_with_model(generator_model, latent_dimension, num_samples):
    x_input = generate_latent_points(latent_dimension, num_samples)
    X = generator_model.predict(x_input)
    y = np.zeros((num_samples, 1))
    return X, y

def generate_latent_points(latent_dimension, num_samples):
    x_input = randn(latent_dimension * num_samples)
    x_input = x_input.reshape(num_samples, latent_dimension)
    return x_input

def train_discriminator(model, dataset, num_iterations=100, num_batches=128):
    half_batch = int(num_batches/2)
    for i in range (num_iterations):
        # get randomly selected 'real' samples
        X_real, y_real = generate_real_samples(dataset, half_batch)
        # update discriminator on real samples
        _, real_acc = model.train_on_batch(X_real, y_real)
		# generate 'fake' examples
        X_fake, y_fake = generate_fake_samples(half_batch, False)
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

def train_complete_model(generator_model, 
discriminator_model, 
gan_model, 
dataset, 
latent_dimension, 
num_epochs=1500, 
batch_size=32):
    batches_per_epoch = int(dataset.shape[0] / batch_size)
    half_batch = int(batch_size / 2)
    for i in range(num_epochs):
        for j in range(batches_per_epoch):
            X_real, y_real = generate_real_samples(dataset, half_batch)
            X_fake, y_fake = generate_fake_samples_with_model(generator_model, latent_dimension, half_batch)
            X, y = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))
            discriminator_loss, _ = discriminator_model.train_on_batch(X, y)
            X_gan = generate_latent_points(latent_dimension, batch_size)
            y_gan = np.ones((batch_size, 1))
            generator_loss = gan_model.train_on_batch(X_gan, y_gan)
            print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, batches_per_epoch, discriminator_loss, generator_loss))
            if(i + 1) % 100 == 0:
                print('Summary----------------------------------------')
                performance_summary(i, generator_model, discriminator_model, dataset, latent_dimension)
                filename = 'generator_model_%03d.h5' % (i + 1)
                generator_model.save(filename)


# load the dataset
dataset = loadtxt('diabetes.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]
y = np.reshape(y,(np.size(y), 1))
x_scaled = minmax_scale(dataset[:,0:8])
scaled_dataset = np.hstack((x_scaled,y))
print('First 5 scaled rows: ')
print(scaled_dataset[:5])

print('Scaled', scaled_dataset.shape)

print('Train', x_scaled.shape, y.shape)
print('Test', X.shape, y.shape)

discriminator_model = discriminator()

train_discriminator(discriminator_model,scaled_dataset)

latent_dimension = 3
generator_model = generator(latent_dimension)

gan_model = build_gan(generator_model, discriminator_model)

gan_model.summary()

train_complete_model(generator_model, discriminator_model, gan_model, scaled_dataset, latent_dimension)

#print(generate_fake_samples_with_model(generator_model,50,10))

#train_discriminator(discriminator_model, dataset)

#fakes = generate_fake_samples(10,False)
#reals = generate_real_samples(X,10)


