from sklearn.metrics import precision_recall_curve, plot_precision_recall_curve, roc_curve, plot_roc_curve
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from lightgbm import LGBMClassifier
from numpy import loadtxt
from os import chdir
from matplotlib import pyplot as plt
pima = loadtxt('./data/diabetes.csv', delimiter=',')
x = pima[:, 0:8]
y = pima[:, 8]
train_x, test_x, train_y, test_y = train_test_split(x, 
                                                    y,
                                                    test_size = 0.2,
                                                    random_state = 17,
                                                    stratify=y)
lgbc = LGBMClassifier(n_estimators=90, 
                          random_state = 94, 
                          max_depth=4,
                          num_leaves=15,
                          objective='binary',
                          metrics ='auc')
pima_model = lgbc.fit(train_x, train_y)
prob1 = pima_model.predict_proba(test_x)[:, 1]
precision, recall, thresholds = precision_recall_curve(test_y, prob1)
plt.plot(recall, precision)
prcplot = plot_precision_recall_curve(pima_model, test_x, test_y)
prcplot.ax_.set_title('2-class Precision-Recall curve: ')
plt.show(block=True)
fpr, tpr, threshold = roc_curve(test_y, prob1)
plt.plot(fpr, tpr)
plot_roc_curve(pima_model, test_x, test_y)