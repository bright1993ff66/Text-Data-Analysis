# Date: 2018 - 06 - 18
# Author: Haoliang Chang
# Data: The data could be found in here：http://thuctc.thunlp.org/
import time
import numpy as np
from collections import Counter

from scipy.sparse import coo_matrix
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

from imblearn.under_sampling import NearMiss
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.metrics import classification_report_imbalanced


# print_results function helps us output the metrics for the model evaluation
def print_results(headline, true_value, pred):
    print(headline)
    print("accuracy: {}".format(metrics.accuracy_score(true_value, pred)))
    print("precision: {}".format(metrics.precision_score(true_value, pred, average='micro')))
    print("recall: {}".format(metrics.recall_score(true_value, pred, average='micro')))
    print("f1: {}".format(metrics.f1_score(true_value, pred, average='micro')))


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

# Just a warm up: Use the logistics regression to complete this Chinese document classification task
# We first use the training data to train the model and apply the model to the validation data
print('===============================Without Undersampling Starts===============================')
print('Logistics Regression starts(validation).....')

start = time.time()

classifier = LogisticRegression
pipeline = make_pipeline(classifier(C=40, random_state=0))
multiC = OneVsRestClassifier(estimator=pipeline)
validation_result = multiC.fit(X_train, y_train).predict(X_validation)
true_validation = np.array(y_validation)

end = time.time()

print('Total time - Without Undersampling: ', end - start, ' seconds\n')
print(metrics.classification_report(y_validation, validation_result))
print()
print('Without Undersampling -  Pipeline Score {}'.format(multiC.fit(X_train, y_train).score(X_validation, y_validation)))
print()
print_results("Without Undersampling - Validation set: ", true_validation, validation_result)

# Predict the test data
print('\nLogistics Regression starts(test).....')

result_test = multiC.fit(X_train, y_train).predict(X_test)
true_test = np.array(y_test)

print_results("Without Undersampling - Test set: ", true_test, result_test)

print('===============================Without Undersampling Ends===============================\n')

print('===============================With Undersampling Starts===============================\n')

start = time.time()

# build model with undersampling
nearmiss_pipeline = make_pipeline_imb(NearMiss(random_state=0), multiC)
nearmiss_model = nearmiss_pipeline.fit(X_train, y_train)
nearmiss_prediction = nearmiss_model.predict(X_validation)

end = time.time()

# Print the distribution of labels about both models
print()
print("Without Undersampling - data distribution: {}".format(Counter(y_train)))
X_nearmiss, y_nearmiss = NearMiss(random_state = 0).fit_sample(X_train, y_train)
print("With Undersampling - data distribution: {}".format(Counter(y_nearmiss)))
print()

# Here comes the result with Undersampling
print('Total time - With Undersampling: ', end - start, ' seconds\n')
print(classification_report_imbalanced(y_validation, nearmiss_prediction))
print()
print('NearMiss Pipeline Score {}'.format(nearmiss_pipeline.score(X_validation, y_validation)))
print()
print_results("NearMiss classification", y_validation, nearmiss_prediction)

print('===============================With Undersampling Ends===============================\n')
