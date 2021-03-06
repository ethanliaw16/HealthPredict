import lightgbm as lgb
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, average_precision_score,precision_recall_curve, plot_precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, auc
from matplotlib import pyplot as plt
#from ctgan import CTGANSynthesizer


data = pd.read_csv('./data/all_data_important_columns.csv')
print(data['State'].value_counts())
synthesized_data = pd.read_csv('./data/ctgan_generated_data.csv')

data.rename(columns={'Smoker.x': 'Smoke'},inplace = True)

encoded_gender = pd.get_dummies(data.Gender, prefix='Gender')
encoded_gender.reset_index(drop=True)

encoded_smoking = pd.get_dummies(data.Smoke, prefix='Smoke')
encoded_smoking.reset_index(drop=True)
data.reset_index(drop=True)
data = pd.concat([data,encoded_smoking], axis=1)
data = pd.concat([data,encoded_gender], axis=1)

encoded_gender_synthesized = pd.get_dummies(synthesized_data.Gender, prefix='Gender')
encoded_gender_synthesized.reset_index(drop=True)

encoded_smoking_synthesized = pd.get_dummies(synthesized_data.Smoke, prefix='Smoke')
encoded_smoking_synthesized.reset_index(drop=True)
synthesized_data.reset_index(drop=True)
synthesized_data = pd.concat([synthesized_data,encoded_smoking_synthesized], axis=1)
synthesized_data = pd.concat([synthesized_data, encoded_gender_synthesized], axis = 1)
print('gan data after encoding smoking: ', synthesized_data.shape)
y = data[['DMIndicator']]

data['L2_HypertensionEssential'] = data['L2_HypertensionEssential'].clip(0,1)
data['L2_MixedHyperlipidemia'] = data['L2_MixedHyperlipidemia'].clip(0,1)
data['L2_ChronicRenalFailure'] = data['L2_ChronicRenalFailure'].clip(0,1)
data['L2_Alcohol'] = data['L2_Alcohol'].clip(0,1)
data['L2_Hypercholesterolemia'] = data['L2_Hypercholesterolemia'].clip(0,1)
data['L2_AtherosclerosisCoronary'] = data['L2_AtherosclerosisCoronary'].clip(0,1)
data['L2_HyperlipOther'] = data['L2_HyperlipOther'].clip(0,1)

X_dataframe = data[[ 
'YearOfBirth',
'Gender_M', 
'HeightMedian', 
'WeightMedian', 
'BMIMedian', 
'SystolicBPMedian', 
'DiastolicBPMedian', 
'L2_HypertensionEssential', 
'L2_MixedHyperlipidemia', 
'L2_ChronicRenalFailure', 
'L2_Alcohol', 
'L2_Hypercholesterolemia', 
'L2_AtherosclerosisCoronary',
'L2_HyperlipOther',
'Smoke_2', 
'Smoke_15', 
'Smoke_20', 
'Smoke_30', 
'Smoke_40']]

Synthesized_X = synthesized_data[[
'YearOfBirth',
'Gender_M', 
'HeightMedian', 
'WeightMedian', 
'BMIMedian', 
'SystolicBPMedian', 
'DiastolicBPMedian', 
'L2_HypertensionEssential', 
'L2_MixedHyperlipidemia', 
'L2_ChronicRenalFailure', 
'L2_Alcohol', 
'L2_Hypercholesterolemia', 
'L2_AtherosclerosisCoronary',
'L2_HyperlipOther',
'Smoke_2', 
'Smoke_15', 
'Smoke_20', 
'Smoke_30', 
'Smoke_40']]
#Synthesized_X = Synthesized_X.rename(columns={"Smoking_2": "Smoke_2", "Smoking_15": "Smoke_15", "Smoking_20": "Smoke_20", "Smoking_30": "Smoke_30", "Smoking_40": "Smoke_40",})
synthesized_y = synthesized_data['DMIndicator']

X_diabetic = data.loc[data['DMIndicator']==1]
X_nondiabetic = data.loc[data['DMIndicator']==0]

y_diabetic = X_diabetic['DMIndicator']
y_nondiabetic= X_nondiabetic['DMIndicator']

x_diabetic_input = X_diabetic[['YearOfBirth', 
'Gender_M',
'HeightMedian', 
'WeightMedian', 
'BMIMedian', 
'SystolicBPMedian', 
'DiastolicBPMedian', 
'L2_HypertensionEssential', 
'L2_MixedHyperlipidemia', 
'L2_ChronicRenalFailure', 
'L2_Alcohol', 
'L2_Hypercholesterolemia', 
'L2_AtherosclerosisCoronary',
'L2_HyperlipOther',
'Smoke_2', 
'Smoke_15', 
'Smoke_20', 
'Smoke_30', 
'Smoke_40']]

x_nondiabetic_input = X_nondiabetic[[ 
'YearOfBirth', 
'Gender_M',
'HeightMedian', 
'WeightMedian', 
'BMIMedian', 
'SystolicBPMedian', 
'DiastolicBPMedian', 
'L2_HypertensionEssential', 
'L2_MixedHyperlipidemia', 
'L2_ChronicRenalFailure', 
'L2_Alcohol', 
'L2_Hypercholesterolemia', 
'L2_AtherosclerosisCoronary',
'L2_HyperlipOther',
'Smoke_2', 
'Smoke_15', 
'Smoke_20', 
'Smoke_30', 
'Smoke_40']]

x_nondiabetic_small = x_nondiabetic_input
y_nondiabetic_small = y_nondiabetic

complete_X = pd.concat([x_nondiabetic_small, x_diabetic_input])
complete_y = pd.concat([y_nondiabetic_small, y_diabetic])

weight = 20 * complete_y + 1
complete_X['Class_Weight'] = weight

complete_X_with_synthesized = pd.concat([x_nondiabetic_small, x_diabetic_input, Synthesized_X])
complete_y_with_synthesized = pd.concat([y_nondiabetic_small, y_diabetic, synthesized_y])

train_X,test_X,train_y,test_y = train_test_split(complete_X, complete_y, random_state=1029328)
test_X['Class_Weight'].values[:]=10.5
print('shape of training data combined with synthesized data ', complete_X_with_synthesized.shape)
#train_X = pd.concat([train_X, Synthesized_X])
#train_y = pd.concat([train_y, synthesized_y])
print('shape of training labels combined with synthesized labels ', complete_y_with_synthesized.shape)

train_X_synthesized = pd.concat([train_X, Synthesized_X])
train_y_synthesized = pd.concat([train_y, synthesized_y])
print('Synthesized X shape ', train_X.shape)
print('Synthesized Y shape ', train_y.shape)
print('First 10 rows of synthesized x ', Synthesized_X[:10])

lgb_train = lgb.Dataset(train_X, train_y)
lgb_eval = lgb.Dataset(test_X, test_y, reference=lgb_train)

# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'auc'},
    'num_leaves': 70,
    'is_unbalance':True,
    #'num_iterations': 5000,
    'learning_rate': 0.001,
    'feature_fraction': 0.4,
    'bagging_fraction': 0.5,
    'bagging_freq': 1,
    'verbose': 0,
    'weight_column':'Class_Weight'
}

print('Starting training...')
# train
#gbm = lgb.cv(params,
#            lgb_train,
#            num_boost_round=1000,
#            metrics='auc',
#            early_stopping_rounds=300,
#            verbose_eval=True)
#print('Cross validation results', gbm)
gbm_trained = lgb.train(params,
                        lgb_train,
                        num_boost_round=500,
                        valid_sets=lgb_eval)

gbm_classifier = LGBMClassifier(n_estimators=90, 
                          random_state = 94, 
                          max_depth=4,
                          num_leaves=31,
                          objective='binary',
                          metrics ='auc')

gbm_classifier_synthesized = LGBMClassifier(n_estimators=90, 
                          random_state = 94, 
                          max_depth=4,
                          num_leaves=31,
                          objective='binary',
                          metrics ='auc')

#Create separate training set containing gan-synthesized datapoints



#train/test one classifier on original data and the other on syntheszied data, test both with original unsynthesized test set
#classifier_model = gbm_classifier.fit(train_X, train_y)
#classifier_probs = classifier_model.predict_proba(test_X)[:,1]
#
#classifier_model_synthesized = gbm_classifier_synthesized.fit(train_X_synthesized,train_y_synthesized)
#classifier_probs_synthesized = classifier_model_synthesized.predict_proba(test_X)[:,1]
#
#
#precision, recall, thresholds = precision_recall_curve(test_y, classifier_probs)
#precision_s, recall_s, thresholds_s = precision_recall_curve(test_y, classifier_probs_synthesized)
#plt.plot(recall, precision, label='Original Training Data Only')
#plt.plot(recall_s,precision_s, label='Original and Synthesized Data Combined')
#plt.legend()
#plt.title('Precision Recall Curve of Model using only original Data vs Original and Synthesized')
#plt.show(block=True)
#
#fpr,tpr,threshold = roc_curve(test_y, classifier_probs)
#fpr_s, tpr_s, threshold = roc_curve(test_y, classifier_probs_synthesized)
#
#print('Area under curve for original: ', auc(fpr,tpr))
#print('Area under curve for synthesized: ', auc(fpr_s, tpr_s))
#
#plt.plot(fpr,tpr, label='Original Training Data Only')
#plt.plot(fpr_s, tpr_s, label='Original and Synthesized Data Combined')
#plt.legend()
#plt.show(block=True)
print('Saving model...')
# save model to file
#gbm.save_model('model.txt')

filename = './trained_models/gbm_predictor.txt'
pickle.dump(gbm_trained, open(filename,'wb'))

loaded_gbm = pickle.load(open(filename, 'rb'))
print('Starting predicting...')
# predict
y_pred = loaded_gbm.predict(test_X, num_iteration=loaded_gbm.best_iteration)
fpr,tpr,thresholds = roc_curve(test_y, y_pred)

differences = abs(tpr - (1 - fpr))
threshold = thresholds[np.argmin(differences)]
print('threshold to use: ', threshold)
pred_max = np.max(y_pred)
pred_min = np.min(y_pred)
min_max_and_threshold = np.array([pred_min,threshold,pred_max])
np.savetxt('./data/gbm_scalers.csv', min_max_and_threshold, delimiter=',')

print(np.max(y_pred))
print(np.min(y_pred))
single_pred = y_pred[0]
if(single_pred >= threshold):
    scaled_pred = ((single_pred - threshold) / (2 * (pred_max - threshold))) + .5
else:
    scaled_pred = ((single_pred - pred_min) / (2 * (threshold - pred_min)))

print('First prediction scaled to 0 and 1: ', scaled_pred)
y_rounded = (y_pred > threshold) * 1


#default_input = [[1990, 67, 180, 28.2, 120, 80, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,2.5]]
#single_pred = loaded_gbm.predict(default_input, num_iteration=loaded_gbm.best_iteration, raw_score=True)
#print('Chance of type 2 Diabetes: %03f' % (y_pred[0]))

plt.plot(thresholds,1-fpr,label='threshold vs fpr')
plt.plot(thresholds,tpr,label='threshold vs tpr')
plt.legend()
plt.show(block=True)
plt.plot(fpr,tpr,label='tpr vs fpr')
plt.show(block=True)
#Y_pred = np.argmax(y_pred, axis=0)
# eval
#print('The rmse of prediction is:', mean_squared_error(test_y, y_pred) ** 0.5)
print('Confusion Matrix\n',confusion_matrix(test_y,y_rounded))
#print('Average Precision Recall ', average_precision_score(test_y, y_pred))
#plot_precision_recall_curve(classifier_model, test_X, test_y)

