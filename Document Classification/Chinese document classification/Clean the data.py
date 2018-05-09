# Date: 2018 - 05 - 09
# Author: Haoliang Chang
# Data: The data could be found in here：http://thuctc.thunlp.org/

import numpy as np
import re

def raw_to_words(raw, remove_stopwords = False):
    
    #0. Get the stopwords from a local file
    stopwords_from_file = open('F:\Data Analysis\github\THUCNews\data\stopwords.txt','r').readlines()
    stopwords = [re.sub('  \n','',char) for char in stopwords_from_file]
    
    #1. Remove non-Chinese-words
    Chinese_only = re.sub(u'[\u3000\n]', u'', raw)
    
    #2. Remove all punctuations
    Chinese_without_punctuations = re.sub(u'[\，\?\、\。\“\”\《\》\！\：\；\？\ ]',u'', Chinese_only)
    
    #3. Remove stopwords
    if remove_stopwords:
        stopwords = set(stopwords)
        result = [word for word in Chinese_without_punctuations if word not in stopwords]
    else:
        result = Chinese_without_punctuations
        
    return result
