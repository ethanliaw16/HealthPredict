from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, Dropout
from keras.optimizers import Adam
from numpy.random import randint, uniform, randn, rand
from sklearn.preprocessing import minmax_scale
import numpy as np
import random

dataset = loadtxt('diabetes.csv', delimiter=',')