import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')# neglect the warnings

import logging
import os.path
import sys
import multiprocessing

from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


if __name__ == '__main__':

    
    # Get the word vectors using gensim
    fdir = r'F:/Data Analysis/github/THUCNews/'
    
    program = fdir + r'Codes/Compute the Chinese word vectors.py'
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # inp is the input corpora, outp1 is the word2vec model, outp2 is the word vectors we created
    inp = fdir + r'data/zhwiki_simple_stemmed.txt'
    outp1 = fdir + r'Codes/wiki.zh.text.model'
    outp2 = fdir + r'data/wiki.zh.text.vector'

    # Skip-gram model
    model = Word2Vec(LineSentence(inp), size=400, window=5, min_count=5,
                     workers=multiprocessing.cpu_count())

    # Save the model and the word vectors
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)

    # Test the model using some examples
    model = gensim.models.Word2Vec.load(fdir + r'Codes/wiki.zh.text.model')

    print(model.most_similar(u'足球'))
    print(model.most_similar(positive = [u'皇上',u'男人'], negative = ['皇后']))
