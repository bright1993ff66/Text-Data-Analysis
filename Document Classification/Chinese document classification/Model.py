# Date: 2018 - 06 - 03
# Author: Haoliang Chang
# Data: The data could be found in here：http://thuctc.thunlp.org/
# Reference: https://www.jianshu.com/p/233da896226a

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import logging
import os.path
import sys
import numpy as np
import pandas as pd
import gensim

def ComputeWordVecs(model, WordList):
    vectors = []
    for word in WordList:
        if word == '\n':
            WordList.remove(word)
        else:
            try:
                vectors.append(model[word])
            except KeyError:
                continue

    return np.array(vectors, dtype = 'float')

def buildVecs(model, filename):
    fileVecs = []
    with open(filename, 'r', encoding='utf-8') as contents:
        for line in contents:
            logger.info("Start line: " + line)
            wordList = line.split(' ')
            vecs = ComputeWordVecs(model, wordList)
            # For each sentence, the mean vector of all its vectors is used to represent this sentence
            if len(vecs) >0:
                vecsArray = sum(np.array(vecs))/len(vecs) # mean
                fileVecs.append(vecsArray)
    return fileVecs 


if __name__ == '__main__':

    fdir = r'F:/Data Analysis/github/THUCNews/'
    
    program = fdir + r'Codes/Model.py'
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # Load the previous word vectors computed from Chinese wiki
    vec_dir = 'data/wiki.zh.text.vector'
    vec_inp = fdir + vec_dir
    model = gensim.models.KeyedVectors.load_word2vec_format(vec_inp,
                                                            binary=False)

    # Load the data
    furniture_inp = fdir + r'data/家居_altogether.txt_stemmed.txt'
    education_inp = fdir + r'data/教育_altogether.txt_stemmed.txt'
    science_inp = fdir + r'data/科技_altogether.txt_stemmed.txt'

    # Build the vectors
    furniture_vecs = buildVecs(model, furniture_inp)
    education_vecs = buildVecs(model, education_inp)
    science_vecs = buildVecs(model, science_inp)

    # Use 0 for furniture, 1 for education, 2 for science
    twos = [2]*len(science_vecs)
    Y = np.concatenate((np.zeros(len(furniture_vecs)),
                        np.ones(len(education_vecs)), np.array(twos)))

    X = furniture_vecs[:]

    for vec in education_vecs:
        X.append(vec)

    for vec in science_vecs:
        X.append(vec)

    X = np.array(X)

    # Write in file   
    df_x = pd.DataFrame(X)
    df_y = pd.DataFrame(Y)
    data = pd.concat([df_y,df_x],axis = 1)
    #print data
    data.to_csv(fdir + 'data/the_data.csv')
