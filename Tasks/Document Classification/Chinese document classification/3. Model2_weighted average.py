# Date: 2018 - 07 - 02
# Author: Haoliang Chang
# Data: The data could be found in here：http://thuctc.thunlp.org/
# Reference: https://openreview.net/pdf?id=SyK00v5xx

import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import logging
import pickle as pkl
import sys
import numpy as np

from mxnet import nd
from mxnet.contrib import text

# Get the word frequency from the dictionary constructed from Chinese wiki
def get_word_frequency(word_text, looktable):
    if word_text in looktable:
        return looktable[word_text]
    else:
        return 1.0

# Get the representation for a file - Weighted Average
def file_to_vec(filename, model, embedding_size = 400, looktable, a=1e-3):
    file_set = []
    with open(filename, 'r', encoding='utf-8') as news:
        contents = news.readlines()
        for new in contents:
            logger.info("Start new: " + new)
            wordlist = new.split(' ')
            vs = np.zeros(embedding_size)  # add all word2vec values into one vector for the sentence
            wordlist_length = len(wordlist)
            for word in wordlist:
                word_vec = model.get_vecs_by_tokens([word]).asnumpy()
                a_value = a / (a + get_word_frequency(word, looktable))  # smooth inverse frequency, SIF
                vs = np.add(vs, np.multiply(a_value, word_vec))  # vs += sif * word_vector

            vs = np.divide(vs, wordlist_length)  # weighted average
            file_set.append(vs)  # add to our existing re-calculated set of sentences
    return file_set


if __name__ == '__main__':

    fdir = r'F:/Data Analysis/github/THUCNews/'

    program = fdir + r'Codes/Model.py'
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # Load the data
    furniture_inp = fdir + r'data/家居_altogether.txt_stemmed.txt'
    education_inp = fdir + r'data/教育_altogether.txt_stemmed.txt'
    science_inp = fdir + r'data/科技_altogether.txt_stemmed.txt'

    inps = [furniture_inp, education_inp, science_inp]

    freq_table = fdir + r'Codes/my_vocab.pkl'
    with open(freq_table, 'rb') as f:
        my_dict = pkl.load(f)
    print('Dictionary is loaded!')

    # Use the mxnet counter to compute the word embedding
    text_data = ''
    for file in inps:
        text_file = open(file, mode='r', encoding='UTF-8')
        texts = text_file.readlines()
        text_for_this_file = ''.join(texts)
        text_data += text_for_this_file

    counter = text.utils.count_tokens_from_str(text_data, token_delim=' ')
    my_vocab = text.vocab.Vocabulary(counter)
    my_embedding = text.embedding.create('fasttext',
                                         pretrained_file_name='wiki.zh.vec', vocabulary=my_vocab)

    """
    # Load the model trained from the wiki Chinese from scratch
    model = gensim.models.Word2Vec.load(fdir + r'Codes/wiki.zh.text.model')
    """

    # Build the vectors
    furniture_vecs = file_to_vec(furniture_inp, my_embedding, embedding_size = 400, looktable = my_dict, a=1e-3)
    education_vecs = file_to_vec(furniture_inp, my_embedding, embedding_size = 400, looktable = my_dict, a=1e-3)
    science_vecs = file_to_vec(furniture_inp, my_embedding, embedding_size = 400, looktable = my_dict, a=1e-3)

    # Use 0 for furniture, 1 for education, 2 for science
    twos = [2] * len(science_vecs)
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
