import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def modelAccuracy():
    #Loading the heart csv data
    dfTest = pd.read_csv('heart.csv')

    #Gets the actual target values
    act = []
    actual = np.asarray(dfTest)
    for i in range(len(actual)):
        act.append(int((actual[i][-1]).item()))

    #Creates the dataset for the testing data
    testData = np.asarray(dfTest.drop('target',1))

    #gets the mean and standard deviation values and scales them to be a value 0 < value < 1
    means = np.mean(testData, axis=0)
    stds = np.std(testData, axis=0)
    testData = (testData - means)/stds

    #Tests the model on a sample data set
    pkl_filename = "heart_disease_model.pkl"
    heartDiseaseLoadedModel = pickle.load(open(pkl_filename, 'rb'))
    prediction = heartDiseaseLoadedModel.predict(testData)
    prediction = prediction.tolist()

    # actual values
    actual = act
    # predicted values
    predicted = prediction

    # confusion matrix
    matrix = confusion_matrix(actual,predicted, labels=[1,0])
    print('Confusion matrix : \n',matrix)

    # outcome values order in sklearn
    tp, fn, fp, tn = confusion_matrix(actual,predicted,labels=[1,0]).reshape(-1)
    print('Outcome values : \n', tp, fn, fp, tn)

    # classification report for precision, recall f1-score and accuracy
    matrix = classification_report(actual,predicted,labels=[1,0])
    print('Classification report : \n',matrix)
