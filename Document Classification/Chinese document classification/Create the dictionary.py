# Date: 2018 - 06 - 24
# Author: Haoliang Chang
# Data: The data could be found in here：http://thuctc.thunlp.org/

import sys
import logging
import pickle # Use pickle to save the dictionary into a local file
from mxnet.contrib import text

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

    # Use the mxnet counter to compute the word embedding
    text_data = ''
    for file in inps:
        text_file = open(file, mode='r', encoding='UTF-8')
        texts = text_file.readlines()
        text_for_this_file = ''.join(texts)
        text_data += text_for_this_file

    text_data.encode('UTF-8')
    counter = text.utils.count_tokens_from_str(text_data, token_delim=' ')
    my_vocab = dict(counter)

    output_dir = open(fdir + r'Codes/my_vocab.pkl', mode = 'wb+')
    pickle.dump(my_vocab, output_dir)

    output_dir.close()



