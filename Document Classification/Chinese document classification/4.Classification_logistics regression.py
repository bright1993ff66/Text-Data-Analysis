# Date: 2018 - 06 - 18
# Author: Haoliang Chang
# Data: The data could be found in here：http://thuctc.thunlp.org/
# Reference: https://www.jianshu.com/p/233da896226a

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import time
import numpy as np

from scipy.sparse import coo_matrix
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

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

# Save the training data, validation data and the test data
outpdir = r'F:\Data Analysis\github\THUCNews\data'
np.save(outpdir + r'\X_train', X_train)
np.save(outpdir + r'\X_validation', X_validation)
np.save(outpdir + r'\X_test', X_test)
np.save(outpdir + r'\y_train', y_train)
np.save(outpdir + r'\y_validation', y_validation)
np.save(outpdir + r'\y_test', y_test)

# Just a warm up: Use the logistics regression to complete this document classification task
print('Logistics Regression starts(validation).....')

start = time.time()

lr = LogisticRegression(C=40, random_state=0)
multiC = OneVsRestClassifier(estimator=lr)
validation_result= multiC.fit(X_train, y_train).predict(X_validation)
true_validation = np.array(y_validation)

# Calculating the precision score for the validation set
precision_lr_validation = metrics.precision_score(true_validation, validation_result, average = 'micro')

# Calculating the F1 score for the test set
f1_score_validation = metrics.f1_score(true_validation, validation_result, average = 'micro')

end = time.time()

print('Total time for training is: ', end - start, ' seconds\n')
print('The f1 score for the validation set is: ', f1_score_validation, '\n')
print('The precision score is: ', precision_lr_validation)

# Predict the test set
print('\nLogistics Regression starts(test).....')

result_test = multiC.fit(X_train, y_train).predict(X_test)
true_test = np.array(y_test)

precision_lr_test = metrics.precision_score(true_test, result_test, average = 'micro')

f1_score_test= metrics.f1_score(true_test, result_test, average = 'micro')

print('The f1 score for the test set is: ', f1_score_test, '\n')
print("The precision score for the test set is: ", precision_lr_test)



