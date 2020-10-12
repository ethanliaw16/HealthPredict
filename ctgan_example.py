from ctgan import CTGANSynthesizer
import pandas as pd
import numpy as np

data = pd.read_csv('./data/all_data_important_columns.csv')

data['L2_HypertensionEssential'] = data['L2_HypertensionEssential'].clip(0,1)
data['L2_MixedHyperlipidemia'] = data['L2_MixedHyperlipidemia'].clip(0,1)
data['L2_ChronicRenalFailure'] = data['L2_ChronicRenalFailure'].clip(0,1)
data['L2_Alcohol'] = data['L2_Alcohol'].clip(0,1)
data['L2_Hypercholesterolemia'] = data['L2_Hypercholesterolemia'].clip(0,1)
data['L2_AtherosclerosisCoronary'] = data['L2_AtherosclerosisCoronary'].clip(0,1)
data['L2_HyperlipOther'] = data['L2_HyperlipOther'].clip(0,1)
data.rename(columns={'Smoker.x': 'Smoke'},inplace = True)
dm_positive = data.loc[data['DMIndicator']==1]

discrete_columns = ['DMIndicator',
'Gender',
'State', 
'L2_HypertensionEssential',
'L2_MixedHyperlipidemia',
'L2_ChronicRenalFailure',
'L2_Alcohol',
'L2_Hypercholesterolemia',
'L2_AtherosclerosisCoronary',
'L2_HyperlipOther',
'Smoke']

ctgan = CTGANSynthesizer()
ctgan.fit(dm_positive, discrete_columns)
#ctgan.save('./trained_models')
samples = ctgan.sample(2000)
print('First 10 generated samples: ', samples[:10])
samples_dm = samples['DMIndicator'].to_numpy()
dm_rounded = np.rint(samples_dm.astype(np.double))
print(np.unique(dm_rounded, return_counts=True))
print(samples['Smoke'].value_counts())
print(samples['Gender'].value_counts())
print('Max height', samples['HeightMedian'].max())
print('Min height', samples['HeightMedian'].min())
print('Year of birth average ', samples['YearOfBirth'].mean())

samples.to_csv('./data/ctgan_generated_data.csv', index=False)