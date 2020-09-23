import lightgbm as lgb
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv('./data/all_data_important_columns.csv')
data.rename(columns={'Smoker.x': 'Smoke'},inplace = True)
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
'Smoke']]

X_diabetic = data.loc[data['DMIndicator']==1]
X_nondiabetic = data.loc[data['DMIndicator']==0]

y_diabetic = X_diabetic['DMIndicator']
y_nondiabetic= X_nondiabetic['DMIndicator']

x_diabetic_input = X_diabetic[['YearOfBirth', 
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
'Smoke']]

x_nondiabetic_input = X_nondiabetic[[ 
'YearOfBirth', 
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
'Smoke']]

x_nondiabetic_small = x_nondiabetic_input[:2000]
y_nondiabetic_small = y_nondiabetic[:2000]

complete_X = pd.concat([x_nondiabetic_small, x_diabetic_input])
complete_y = pd.concat([y_nondiabetic_small, y_diabetic])

train_X,test_X,train_y,test_y = train_test_split(complete_X, complete_y, random_state=42)

lgb_train = lgb.Dataset(train_X, train_y)
lgb_eval = lgb.Dataset(test_X, test_y, reference=lgb_train)

# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'auc'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbose': 0
}

print('Starting training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval)

print('Saving model...')
# save model to file
#gbm.save_model('model.txt')

filename = './trained_models_gbm_predictor.txt'
pickle.dump(gbm, open(filename,'wb'))

loaded_gbm = pickle.load(open(filename, 'rb'))
print('Starting predicting...')
# predict
y_pred = loaded_gbm.predict(test_X, num_iteration=gbm.best_iteration)
y_rounded = np.rint(y_pred)
print(y_rounded[:10])
#Y_pred = np.argmax(y_pred, axis=0)
# eval
#print('The rmse of prediction is:', mean_squared_error(test_y, y_pred) ** 0.5)
print('Confusion Matrix\n',confusion_matrix(test_y,y_rounded))


