# Date: 2018 - 05 - 08
# Author: Haoliang Chang
# Data: The data could be found in here：http://thuctc.thunlp.org/

import os
import glob
import errno
import numpy as np

# Get the data from multiple txt files
def get_data(path):
    files = glob.glob(path)
    data = []

    for name in files:
        try:
            with open(name, encoding='UTF-8') as f:
                data.append([f.read()])
        except IOError as exc:  # Not sure what error this is
            if exc.errno != errno.EISDIR:
                raise

    return data

path_furniture = 'F:\Data Analysis\github\THUCNews\家居\*.txt'
path_education = 'F:\Data Analysis\github\THUCNews\教育\*.txt'
path_science = 'F:\Data Analysis\github\THUCNews\科技\*.txt'

furniture = get_data(path_furniture)
education = get_data(path_education)
science = get_data(path_science)

# The shape of these data
print(np.shape(furniture))
print(np.shape(education))
print(np.shape(science))


