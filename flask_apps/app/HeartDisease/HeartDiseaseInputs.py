import HeartDisease
import ResetData
import numpy as np
from numpy import loadtxt
import pickle

def testModelWithInputs(input):
    pkl_filename = "heart_disease_model.pkl"
    modelAccuracy = 0
    loadedAccuracy = 0
    while(modelAccuracy < 0.8 and loadedAccuracy < 0.1 ):
        try:
            heartDiseaseLoadedModel = pickle.load(open(pkl_filename, 'rb'))
            loadedAccuracy = heartDiseaseLoadedModel.score(testData, testLabel)
        except:
            modelAccuracy = HeartDisease.createAndLoadModel()

    loaded_means = loadtxt('heart_disease_means.csv')
    loaded_stds = loadtxt('heart_disease_stds.csv')
    sampleInputs = np.asarray(input)
    sampleInputs = (sampleInputs - loaded_means) / loaded_stds
    sampleInputs = sampleInputs.reshape(1,-1)

    predictionProbability = heartDiseaseLoadedModel.predict_proba(sampleInputs)
    prediction = heartDiseaseLoadedModel.predict(sampleInputs)

    if(prediction == 1):
        return (1,predictionProbability[0][1])
    else:
        return (0,predictionProbability[0][0])
    # Line to print the probability that the person has/doesn't have heart disease print('Probability:', predictionProbability)
    # Line to print if the person has heart disease or not print('prediction:', prediction)


if __name__ == '__main__':
    #sampleFeature = [ 46, 1, 1, 101, 197, 1, 1, 156, 0, 0, 2, 0, 3]
    sampleFeature = [ 20, 0, 1, 120, 169, 0, 1, 180, 1, 1, 1, 0, 3]

    value = testModelWithInputs(sampleFeature)
    print(value[0])
    print(value[1])
