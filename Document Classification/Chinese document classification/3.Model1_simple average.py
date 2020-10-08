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
import re

from mxnet import nd
from mxnet.contrib import text

# Compute the word vector for each word in a word list
def ComputeWordVecs(model, WordList):
    vectors = []
    for word in WordList:
        word = word.replace('\n','')
        try:
            vec = model.get_vecs_by_tokens([word]).asnumpy()
            vectors.append(vec)
        except:
            continue
    return vectors

# Get the representation for a file
def buildVecs(model, filename):
    fileVecs = []
    with open(filename, 'r', encoding='utf-8') as news:
        contents = news.readlines()
        for new in contents:
            logger.info("Start new: " + new)
            wordList = new.split(' ')
            vecs = ComputeWordVecs(model, wordList)
            # For each sentence, the mean vector of all its vectors is used to represent this sentence
            if len(vecs) >0:
                vecsArray = np.sum(vecs, axis = 0)/len(vecs) # mean
                fileVecs.append(vecsArray)
    return fileVecs


if __name__ == '__main__':

    fdir = r'F:/Data Analysis/github/THUCNews/'
    
    program = fdir + r'Codes/Model.py'
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # Load the data
    furniture_inp = fdir + r'data/家居_altogether.txt_stemmed.txt'
    education_inp = fdir + r'data/教育_altogether.txt_stemmed.txt'
    science_inp = fdir + r'data/科技_altogether.txt_stemmed.txt'

    inps = [furniture_inp, education_inp, science_inp]

    # Use the mxnet counter to compute the word embedding
    text_data = ''
    for file in inps:
        text_file = open(file, mode = 'r', encoding = 'UTF-8')
        texts = text_file.readlines()
        text_for_this_file = ''.join(texts)
        text_data += text_for_this_file
        
    counter = text.utils.count_tokens_from_str(text_data, token_delim = ' ')
    my_vocab = text.vocab.Vocabulary(counter)
    my_embedding = text.embedding.create('fasttext',
            pretrained_file_name='wiki.zh.vec',vocabulary=my_vocab)

    """
    # Load the model
    model = gensim.models.Word2Vec.load(fdir + r'Codes/wiki.zh.text.model')
    """

    # Build the vectors
    furniture_vecs = buildVecs(my_embedding , furniture_inp)
    education_vecs = buildVecs(my_embedding , education_inp)
    science_vecs = buildVecs(my_embedding , science_inp)

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
    np.save('all_data_X', X)
    np.save('all_data_Y', Y)
