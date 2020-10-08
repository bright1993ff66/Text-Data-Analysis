# Date: 2018 - 05 - 22
# Author: Haoliang Chang
# Data: The data could be found in here：http://thuctc.thunlp.org/
# Reference: https://www.jianshu.com/p/233da896226a

import os
import sys
import logging
import re

# Get all the content of a txt file
def get_data(path):
    f = open(path, mode = 'r', encoding = 'UTF-8')
    lines = f.readlines()
    content = ''.join(lines)
    return content

# Clean the txt data
def raw_to_words(raw, remove_stopwords = False):
    
    #0. Get the stopwords from a local file
    stopwords_from_file = open('F:\Data Analysis\github\THUCNews\data\stopwords.txt','r').readlines()
    stopwords = [re.sub('  \n','',char) for char in stopwords_from_file]
    
    #1. Remove non-Chinese-words
    Words_only = re.sub(u'[\u3000\n\1-9]', u'', raw)
    Chinese_only = re.sub(u'[a-zA-Z]', u'', Words_only)
    
    #2. Remove all punctuations
    Chinese_without_punctuations = re.sub(u'[\，\?\、\。\“\”\《\》\！\：\；\？\-\ ]',u'', Chinese_only)
    
    #3. Remove stopwords
    if remove_stopwords:
        stopwords = set(stopwords)
        without_stopwords = [word for word in Chinese_without_punctuations if word not in stopwords]
        result = ''.join(without_stopwords)
    else:
        result = Chinese_without_punctuations
        
    return result

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format = '%(asctime)s:%(levelname)s:%(message)s')
    logging.root.setLevel(level = logging.INFO)

    # All the inputs
    inputs =  r'F:/Data Analysis/github/THUCNews/'
    folders = ['家居','教育','科技']

    for folder_name in folders:
        logger.info(folder_name + 'is running!')

        outp = folder_name + '_altogether.txt'
        outputs = open(inputs + r'data/' + outp, mode = 'w', encoding = 'UTF-8')

        i = 0
        inpdir = inputs + folder_name

        for parent, dirnames, filenames in os.walk(inpdir):
            for filename in filenames:
                content = get_data(inpdir + r'/' + filename)
                cleaned_content = raw_to_words(content, remove_stopwords = True)
                outputs.writelines(cleaned_content + '\n')
                i += 1

        outputs.close()
        logger.info(str(i) + 'files have saved')

