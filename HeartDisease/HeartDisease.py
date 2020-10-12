import pandas as pd
import numpy as np
from numpy import savetxt, loadtxt
import seaborn as sns
import matplotlib.pyplot as plt
import joblib as jl
from sklearn.linear_model import LogisticRegression
import pickle

def createAndLoadModel():
    #Loading the heart csv data
    heartDF = pd.read_csv('heart.csv')

    #Randomly shifts the data rows in order to randomize the data set
    newHeartDF = heartDF.sample(frac=1)

    #Creates testing, training, and checking subsets from the original data set
    dfTrain = newHeartDF[:243]
    dfTest = newHeartDF[243:253]
    dfCheck = newHeartDF[253:]

    #Creates the labels and datasets for the training and testing data
    trainLabel = np.asarray(dfTrain['target'])
    trainData = np.asarray(dfTrain.drop('target',1))
    testLabel = np.asarray(dfTest['target'])
    testData = np.asarray(dfTest.drop('target',1))

    #gets the mean and standard deviation values and scales them to be a value 0 < value < 1
    means = np.mean(trainData, axis=0)
    stds = np.std(trainData, axis=0)
    trainData = (trainData - means)/stds
    testData = (testData - means)/stds

    #Uses a Logistic Regression in order to create a model of the data
    heartDiseaseCheck = LogisticRegression()
    heartDiseaseCheck.fit(trainData, trainLabel)

    #Checks the accuracy of the model
    accuracy = heartDiseaseCheck.score(testData, testLabel)
    #Line to test the accuracy of the model: print("accuracy = ", accuracy * 100, "%")

    #Creates a file to store the model in
    pkl_filename = "heart_disease_model.pkl"
    storedAccuracy = 0

    #Checks is the heart disease model has already been saved and if not then it
    #saves the current version of the model
    try:
        storedAccuracy = pickle.load(open(pkl_filename , 'rb')).score(testData,testLabel)
    except:
        with open(pkl_filename, 'wb') as file:
            pickle.dump(heartDiseaseCheck, file)
        savetxt('heart_disease_means.csv', means)
        savetxt('heart_disease_stds.csv', stds)

    #Saves the model if it's accuracy is better than 80% and if it's accuracy
    #is better than the previous model
    if accuracy > 0.8 and accuracy > storedAccuracy:
        with open(pkl_filename, 'wb') as file:
            pickle.dump(heartDiseaseCheck, file)
        savetxt('heart_disease_means.csv', means)
        savetxt('heart_disease_stds.csv', stds)

    #Loads the saved model and then prints its accuracy
    heartDiseaseLoadedModel = pickle.load(open(pkl_filename , 'rb'))
    loaded_means = loadtxt('heart_disease_means.csv')
    loaded_stds = loadtxt('heart_disease_stds.csv')
    loadedAccuracy = heartDiseaseLoadedModel.score(testData,testLabel)
    #Line to test the accuracy of the loaded model print("loaded accuracy = ", loadedAccuracy * 100, "%")

    #Creates the sample data set
    sampleData = dfCheck[2:3]
    #Tests the model on a sample data set
    sampleDataFeatures = np.asarray(sampleData.drop('target',1))
    sampleDataFeatures = (sampleDataFeatures - loaded_means)/loaded_stds
    predictionProbability = heartDiseaseLoadedModel.predict_proba(sampleDataFeatures)
    prediction = heartDiseaseLoadedModel.predict(sampleDataFeatures)
    #Line to show the probability if the patient has heart disease or not print('Probability:', predictionProbability)
    #Line to show the prediction of the loaded model print('prediction:', prediction)

    return loadedAccuracy