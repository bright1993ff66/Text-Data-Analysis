<h1 align="center">Chinese Document Classification</h1>

Here I will do some Chinese document classification based on a real-world dataset.

# 1. Dataset

# 1.1 Documents
The dataset can be found in here [THUCTC](http://thuctc.thunlp.org/), which is made by Natural Language Processing and Computational Social Science Lab, Tsinghua University. The dataset is huge so I will only select three categories of text data(家居,教育,科技) and do the document classification. 
# 1.2 Chinese Stopwords
The common Chinese stopwords can be downloaded in here: [Chinese Stopwords](https://github.com/bright1993ff66/Text-Data-Analysis/blob/master/Document%20Classification/Chinese%20document%20classification/data/stopwords.txt), which is created by Institute of Computing Technology,Chinese Academy of Sciences.

# 2. Data Preprocessing

Before feeding the data to a machine learning model(logistics regression, random forest, SVM, etc.), we should preprocess the data so that our model could read it. Basically, the following steps should be involved:
1. remove the stopwords
2. delete all non-Chinese characters
3. remove all punctuations
4. Chinese segmentation

The following codes could be used to finish these tasks:

```Python
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
    
file_path = r'F:\\Data Analysis\\github\\THUCNews\data\\'
files = ['家居_altogether.txt', '教育_altogether.txt',
             '科技_altogether.txt']

for file in files:
    # Set the input and output files
    file_to_be_segmented = open(file_path + file, mode = 'r', encoding = 'UTF-8')
    output_file = open(file_path + file + '_segmented.txt', mode = 'w', encoding = 'UTF-8')
    print('The file is opened~')
    lineNum = 1
    line = file_to_be_segmented.readline()
        
    # Segment the words using jieba
    while line:
        print('Line(article) ', lineNum, 'is processing')
        segmentation = jieba.cut(line, cut_all = False)
        line_seg = ' '.join(segmentation)
        output_file.writelines(line_seg)
        lineNum += 1
        line = file_to_be_segmented.readline()

        print('finished', file)
        file_to_be_segmented.close()
        output_file.close()
```

# 3. Get the pre-trained word embedding

There are mainly two ways to get the pre-trained word embedding:

1. Compute the pre-trained word embedding from scratch. For this Chinese document classification task, you should first download the wiki Chinese corpora from this link [wiki.zh.vec](https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2) and use the gensim module in Python to train it. This wiki corpora uses the traditional Chinese characters. To know how to transform the transform the traditional Chinese to simplified Chinese and work out the pre-trained word vectors we need, just go to this [page](https://github.com/bright1993ff66/Text-Data-Analysis/tree/master/Document%20Classification/Chinese%20document%20classification/Compute%20the%20word%20vectors%20from%20Chinese%20wiki) and read the codes there.
2. A much more direct way to get the pre-trained word embedding is to use MXNet. MXNet offers hundreds of different word embeddings for various languages. To load the embedding to your model, just use the following code to import the [text API](http://mxnet.incubator.apache.org/api/python/contrib/text.html) in MXNet:

  ```Python
  from mxnet.contrib import text
  ```

  And use the following one line of code to load the pre-trained word embedding:

  ```Python
  my_embedding = text.embedding.create('fasttext', pretrained_file_name='wiki.simple.vec',
  vocabulary=my_vocab)
  ```
  
  For more information, you could go to [MXNet text API official website](http://mxnet.incubator.apache.org/api/python/contrib/text.html) and read more details.
  
# 4. Get the representation of each text file.

In each category of our data(such as '科技'), we could see many text files:

![file_in_category.JPG](https://github.com/bright1993ff66/Text-Data-Analysis/blob/master/Document%20Classification/Chinese%20document%20classification/Pictures/file_in_category.JPG)

In this section, our task now is to work out the representation of each file. Here we use the average of word vectors of all the words in a file as the representation of this file, and we set this method as a baseline. More advanced methods would be used in the future.

The following two functions are used to compute the word representation for each file:

```Python
# Compute the word vector for each word in a word list
def ComputeWordVecs(model, WordList):
    vectors = []
    for word in WordList:
        word = word.replace('\n','')
        try:
            vec = model.get_vecs_by_tokens([word]).asnumpy() # I use fasttext in MXNet. I need to transform the result into numpy array
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
```

5. Classify the documents

Here we first use the logistics regression to do the classification. Since this task is a multi-class classification, we should use the multi-class classification API OneVsRestClassifier in scikit-learn:

```Python
from sklearn.multiclass import OneVsRestClassifier
```
We also notice that the data is also imbalanced, we use the imblearn module to cope with this issue:

```Python
from imblearn.under_sampling import NearMiss
```
Then we compare two models(one with undersampling and another one without undersampling). We both use logistics regression with same hyperparameters. Here is the result without undersampling:

![without undersampling.JPG](https://github.com/bright1993ff66/Text-Data-Analysis/blob/master/Document%20Classification/Chinese%20document%20classification/Pictures/Without%20undersampling.JPG)

And here is the result with undersampling:

![with undersampling.JPG](https://github.com/bright1993ff66/Text-Data-Analysis/blob/master/Document%20Classification/Chinese%20document%20classification/Pictures/With%20undersampling.JPG)

From the results above, I would say that the performance has improved. Because we could see that the recall values of the categories with smaller amount of data increase. Even though the F1 score decreases, I think the model with undersampling is better.
