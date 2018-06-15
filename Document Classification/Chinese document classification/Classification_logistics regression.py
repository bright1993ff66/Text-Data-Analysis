# Date: 2018 - 06 - 14
# Author: Haoliang Chang
# Data: The data could be found in here：http://thuctc.thunlp.org/
# Reference: https://www.jianshu.com/p/233da896226a

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import logging
import os.path
import sys
import time
import numpy as np

from mxnet import nd
from mxnet.contrib import text

from scipy.sparse import coo_matrix
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

from matplotlib import pyplot as plt

# Load the data and label
threed_data = np.load('all_data_X.npy')
label_from_file = np.load('all_data_Y.npy')

twod_data = threed_data.transpose(0,2,1).reshape(len(label_from_file),-1) # Convert a 3D array to a 2D array

# Random shuffle the data and the label
twod_data_sparse = coo_matrix(twod_data)
data, data_sparse, label = shuffle(twod_data, twod_data_sparse, label_from_file, random_state = 0)

# All the text data should be categorized into training set, validation set, test set
X_train_plus, X_test, y_train_plus, y_test = train_test_split(data, label, test_size = 0.2, random_state = 0)
X_train, X_validation, y_train, y_validation = train_test_split(X_train_plus, y_train_plus, test_size = 0.25,
                                                                random_state = 0)


# logistics regression# logis
print('Logistics Regression starts.....')

start = time.time()

lr = LogisticRegression(C=40, random_state=0)
multiC = OneVsRestClassifier(estimator=lr)
result_lr = multiC.fit(X_train, y_train).predict(X_validation)
true = np.array(y_validation)
result_prob_lr = multiC.fit(X_train, y_train).predict_proba(X_validation)[:, 1]

# Calculating the precision score
precision_lr = metrics.precision_score(true, result_lr, average = 'micro')

# Calculating the AUC
fpr_lr, tpr_lr, thresholds_lr = metrics.roc_curve(true, result_prob_lr, pos_label=1)
auc_lr = metrics.auc(fpr_lr, tpr_lr)

end = time.time()

print('Total time for training is: ', end - start, ' seconds\n')
print('The area under the curve is: ', auc_lr)
print('The precision score is: ', precision_lr)

# Draw the RUC curve
plt.plot(fpr_lr, tpr_lr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC Curve for BOWs meet Bag of Popcore - Logistics Regression')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.show()

# Predict the test set
result = multiC.fit(X_train, y_train).predict(X_test)
true = np.array(y_test)
result_prob_lr = multiC.fit(X_train, y_train).predict_proba(X_test)[:, 1]

precision = metrics.precision_score(true, result, average = 'micro')
print("The precision score for the test set is: ", precision)




