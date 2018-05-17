"""
Date: 2018 - 05 - 13
Codes and more specific guidance could be found in this site: 
https://www.jianshu.com/p/ec27062bd453
The zh_wiki and langconv modules, which are used for transforming 
the complex Chinese to the simplified Chinese, could be found in
here: https://github.com/skydark/nstools/tree/master/zhtools
Data: The data could be found in here：http://thuctc.thunlp.org/
"""

# Import the modules
import logging
import os.path
import sys
import zh_wiki
from langconv import *

import jieba
import jieba.analyse
import jieba.posseg as pseg

from gensim.corpora import WikiCorpus

# Export all the Chinese words in the xml file into a txt file
if __name__ == '__main__':
    program = 'F:\Data Analysis\github\THUCNews\Codes\Get Chinese word vectors from wiki.py'
    logger = logging.getLogger(program)
 
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    if len(sys.argv) < 3:
        print(globals()['__doc__'] % locals())
        sys.exit(1)
        
    data_file_path = 'F:\Data Analysis\github\THUCNews\data\zhwiki-latest-pages-articles.xml.bz2'
	output_path = 'F:\Data Analysis\github\THUCNews\data\zhwiki.txt'

    inp, outp = data_file_path, output_path
    space = " "
    i = 0

    output = open(outp, 'w', encoding = 'UTF-8')

    wiki =WikiCorpus(inp, lemmatize=False, dictionary=[]) # The type of the input file should be .xml.bz2
	
    for text in wiki.get_texts():
        output.write(space.join(text) + "\n")
        i = i+1
        if (i % 10000 == 0):
            logger.info("Saved "+str(i)+" articles.")

    output.close()
    logger.info("Finished Saved "+str(i)+" articles.")

	# Transform the complex Chinese to the simplified Chinese
	file_to_be_processed = 'F:\Data Analysis\github\THUCNews\data\zhwiki.txt'
	output_file_path = 'F:\Data Analysis\github\THUCNews\data\zhwiki_simple.txt'
	output = open(output_file_path, 'w', encoding = 'UTF-8')

	i = 0
	
	with open(file_to_be_processed, mode = 'r', encoding = 'UTF-8') as f:
	    for line in f:
	        output.write(cht_to_chs(line))
	        i = i+1
	        if (i % 10000 == 0):
	            logger.info("Saved "+str(i)+" articles.")
	        
	    output.close()
		logger.info("Finished Saved "+str(i)+" articles.")
	
	# Segment the words using jieba
	file_to_be_stemmed = open('F:\Data Analysis\github\THUCNews\data\zhwiki_simple.txt', mode = 'r', encoding = 'UTF-8')
	output_file = open('F:\Data Analysis\github\THUCNews\data\zhwiki_simple.txt', mode = 'w', encoding = 'UTF-8')
	print('The file is opened~')

	lineNum = 1
	line = file_to_be_stemmed.readline()

	while line:
	    print('Line(article) ', lineNum, 'is processing')
	    segmentation = jieba.cut(line, cut_all = False)
	    line_seg = ' '.join(segmentation)
	    output_file.writelines(line_seg)
	    lineNum += 1
	    line = file_to_be_stemmed.readline()
	    
	print('finished')
	file_to_be_stemmed.close()
	output_file.close()
		
