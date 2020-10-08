# Date: 2018 - 05 - 09
# Author: Haoliang Chang
# Data: The data could be found in here：http://thuctc.thunlp.org/

import re

#Get the stopwords from a local file
stopwords_from_file = open('F:\Data Analysis\github\THUCNews\data\stopwords.txt','r').readlines()
stopwords = [re.sub('  \n','',char) for char in stopwords_from_file]
stopwords = set(stopwords)

def raw_to_words(raw, remove_stopwords = False):
    
    #1. Remove non-Chinese-words
    Chinese_only = re.sub(u'[\u3000\n]', u'', raw)
    
    #2. Remove all punctuations
    Chinese_without_punctuations = re.sub(u'[\，\?\、\。\“\”\《\》\！\：\；\？\ ]',u'', Chinese_only)
    
    #3. Stemming
    Chinese_seglist = jieba.cut(Chinese_without_punctuations, cut_all = False)
    seg_sentence = ''
    for word in Chinese_seglist:
        seg_sentence += word + " "
    Chinese_seg = seg_sentence.strip()
    
    
    #4. Remove stopwords
    if remove_stopwords:
        result = [word for word in Chinese_seg if word not in stopwords]
    else:
        result = Chinese_seg
        
    return result
