from nltk.stem.lancaster import LancasterStemmer
import re
import os
import string
import spacy
import zh_core_web_sm
from spacy.lang.en.stop_words import STOP_WORDS

import data_paths

punctuations_all = string.punctuation + '。，﹑？！：；“”（）《》•……【】'

nlp = zh_core_web_sm.load()
traffic_word_set = {'堵', '拥堵', '车祸', '剐蹭', '事故', '绕行', '追尾', '相撞', '塞车', '路况'}


def create_stopwords_set(stopword_path: str) -> list:
    """
    Create the Chinese stopword list
    :param stopword_path: the path which contains the stopword
    :return: a Chinese stopword list
    """
    stopwords_list = []
    with open(os.path.join(stopword_path, 'hit_stopwords.txt'), 'r', encoding='utf-8') as stopword_file:
        for line in stopword_file:
            line = line.replace("\r", "").replace("\n", "")
            stopwords_list.append(line)
    return stopwords_list


def preprocessing(raw_tweet, stemming=False, remove_stopwords=False):
    """
    Preprocess the tweet: consider hashtag words as just words and pass them to linguistic modules
    :param raw_tweet: the tweet text to be processed
    :param stemming: conduct word stemming or not
    :param remove_stopwords: whether remove stopwords or not
    :return: cleaned tweet
    """
    # 0. Remove the urls
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)" \
            r"))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    text_without_url = re.sub(regex, '', raw_tweet)

    # 1. Only consider letters and numbers
    letters_nums_only = re.sub("[^a-zA-Z0-9]", " ", text_without_url)
    #
    # 2. Convert to lower case, split into individual words
    words = letters_nums_only.split()

    # 4. Remove stop words
    if remove_stopwords:
        stops = set(STOP_WORDS)
        meaningful_words = [w for w in words if not w in stops]
    else:
        meaningful_words = words
    #
    # 5. Stemming
    if stemming:
        st = LancasterStemmer()
        text_stemmed = [st.stem(word) for word in meaningful_words]
        result = text_stemmed
    else:
        result = meaningful_words

    return " ".join(result)


def preprocessing_weibo(raw_tweet, return_word_list=False):
    """
    Preprocess the tweet: consider hashtag words as just words and pass them to linguistic modules
    :param raw_tweet: the tweet text to be processed
    :param stemming: conduct word stemming or not
    :param remove_stopwords: whether remove stopwords or not
    :return: cleaned tweet
    """
    # 0. Remove the urls
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)" \
            r"))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    text_without_url = re.sub(regex, '', raw_tweet)

    # 1. Remove the @user patterns
    text_without_username1 = re.sub('@[\u4e00-\u9fa5a-zA-Z0-9_-]{4,30}', '', text_without_url)
    text_without_username2 = re.sub('\/\/@.*?:', '', text_without_username1)

    # 2. Don't consider the numbers, punctuations
    stopwords_list = create_stopwords_set(stopword_path=data_paths.stopword_path)
    words_only = re.sub("[0-9]", "", text_without_username2)
    words_without_puc = re.sub("[{}]".format(punctuations_all), "", words_only)
    #
    # 3. Tokenize the weibos using spacy
    doc = nlp(words_without_puc)
    stopwords_list.append(' ')
    words_list = [token.lemma_ for token in doc]
    words_final = [word for word in words_list if word not in stopwords_list]

    if return_word_list:
        return words_final
    else:
        return ' '.join(words_final)


if __name__ == '__main__':

    sample_text = 'no retweeters'
    result = preprocessing_weibo(raw_tweet=sample_text, return_word_list=True)
    print(result)
