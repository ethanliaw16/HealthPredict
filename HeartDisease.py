import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib as jl
from sklearn.linear_model import LogisticRegression

heartDF = pd.read_csv('heart.csv')

#print(heartDF.head())
#heartDF.info()
#heartDF.info()

#corr = heartDF.corr()
#print(corr)
#sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
#plt.show()

newHeartDF = heartDF.sample(frac=1)

dfTrain = newHeartDF[:243]
dfTest = newHeartDF[243:253]
dfCheck = newHeartDF[253:]

#print(heartDF)
#print(newHeartDF)

trainLabel = np.asarray(dfTrain['target'])
trainData = np.asarray(dfTrain.drop('target',1))
testLabel = np.asarray(dfTest['target'])
testData = np.asarray(dfTest.drop('target',1))

means = np.mean(trainData, axis=0)
stds = np.std(trainData, axis=0)
trainData = (trainData - means)/stds
testData = (testData - means)/stds

diabetesCheck = LogisticRegression()
diabetesCheck.fit(trainData, trainLabel)

#accuracy = diabetesCheck.score(testData, testLabel)
#print("accuracy = ", accuracy * 100, "%")

jl.dump([diabetesCheck, means, stds], 'heartModel.pkl')

diabetesLoadedModel, means, stds = jl.load('heartModel.pkl')
accuracyModel = diabetesLoadedModel.score(testData, testLabel)
print("accuracy = ",accuracyModel * 100,"%")

#print(dfCheck.head())

sampleData = dfCheck[2:3]

sampleDataFeatures = np.asarray(sampleData.drop('target',1))
sampleDataFeatures = (sampleDataFeatures - means)/stds
predictionProbability = diabetesLoadedModel.predict_proba(sampleDataFeatures)
prediction = diabetesLoadedModel.predict(sampleDataFeatures)
print('Probability:', predictionProbability)
print('prediction:', prediction)