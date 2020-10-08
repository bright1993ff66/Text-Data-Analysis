# encoding == 'utf-8'
import os
import re
import zh_core_web_sm
import pandas as pd
import numpy as np
import json
from json.decoder import JSONDecodeError
import csv
from collections import Counter
import pytz

import data_paths
from utils import transform_string_time_to_datetime
from process_text.text_preprocessing import create_stopwords_set

nlp = zh_core_web_sm.load()
# traffic-related word dictionary
traffic_word_set = {'堵', '拥堵', '车祸', '剐蹭', '事故', '绕行', '追尾', '相撞', '塞车', '路况'}
traffic_word_set_update = {'堵', '拥堵', '阻塞','塞车', '车祸', '剐蹭', '事故', '撞', '追尾', '相撞', '路况', '封道',
                           '封路', '绕行'}
# the user id of official traffic informaiton social media accounts in Shanghai
# 1980308627: 乐行上海:
# https://weibo.com/lexingsh?from=page_100206_profile&wvr=6&mod=bothfollow&refer_flag=1005050010_&is_all=1
# 1750349294: 上海交通广播
# https://weibo.com/shjtgb?is_all=1
# 1976304153: 上海市交通委员会
# https://weibo.com/p/1001061976304153/home?from=page_100106&mod=TAB&is_hot=1#place
traffic_acount_uid = {1980308627, 1750349294, 1976304153}
# some selected accounts for analysis
# 2539961154: 上海发布
# https://weibo.com/shanghaicity?topnav=1&wvr=6&topsug=1&is_hot=1
# 1763251165: 宣克灵
# https://weibo.com/xuankejiong?topnav=1&wvr=6&topsug=1&is_hot=1
# 2256231983: 上海热门资讯
# https://weibo.com/209273419?topnav=1&wvr=6&topsug=1&is_hot=1
# 1971537652: 魔都生活圈
# https://weibo.com/u/1971537652?topnav=1&wvr=6&topsug=1&is_hot=1
traffic_acount_uid_final = {2539961154, 1763251165, 2256231983, 1971537652}

timezone_shanghai = pytz.timezone('Asia/Shanghai')

class Chinese_Weibo_Analyze(object):

    """
    Create a Weibo parser
    """

    nlp = zh_core_web_sm.load()

    def __init__(self, weibo: str):

        """
        :param weibo: A Weibo string
        """
        self.weibo = weibo

    @property
    def parse_chinese(self):
        """
        Get the nouns and verbs from a Weibo string
        :return: nouns list and verb list
        """
        regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)" \
                r"))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
        weibo_without_url = re.sub(regex, '', self.weibo)
        doc = Chinese_Weibo_Analyze.nlp(weibo_without_url)
        nouns_list = list(set([token.lemma_ for token in doc if token.pos_ == "NOUN"]))
        verbs_list = list(set([token.lemma_ for token in doc if token.pos_ == "VERB"]))
        stopwords_list = create_stopwords_set(stopword_path=data_paths.chinese_stopword_path)
        nouns_list_without_stopwords = [word for word in nouns_list if word not in stopwords_list]
        verbs_list_without_stopwords = [word for word in verbs_list if word not in stopwords_list]
        return nouns_list_without_stopwords, verbs_list_without_stopwords


def check_weibo_traffic(weibo, weibo_traffic_set) -> bool:
    """
    Get the candidate traffic-related Weibo. The Weibo text should contain at least one keyword
    :param weibo: a Weibo text string
    :param weibo_traffic_set: a traffic-related keyword set
    :return: whether a Weibo string is traffic-related or not
    """
    doc = nlp(str(weibo))
    token_set = set([str(token) for token in doc])
    intersect_words = token_set.intersection(weibo_traffic_set)

    if len(intersect_words) >= 1:
        return True
    else:
        return False


def search_traffic_related_weibos(path, weibo_traffic_dict_set, save_path, processed_file_list=None,
                                  start_counter=None):
    """
    Output the candidate traffic-related Weibos
    :param path: a path containing Weibo dataframes
    :param weibo_traffic_dict_set: a traffic-related word dictionary to get candidate traffic Weibo
    :param save_path: path to save the candidate traffic dataframes
    :param processed_file_list: a list of processed filenames
    :param start_counter: a counter which counts the number of processed files
    :return: None
    """
    result_dataframe_list = []

    if start_counter is not None:
        counter = start_counter
    else:
        counter = 0

    for file in os.listdir(path):

        if processed_file_list is not None:

            if file not in processed_file_list:

                print('Coping with the file: {}'.format(file))
                dataframe = pd.read_csv(os.path.join(path, file), encoding='utf-8', index_col=0)
                dataframe_select = dataframe[dataframe.apply(lambda row: check_weibo_traffic(
                    row['text'], weibo_traffic_set=weibo_traffic_dict_set) or check_weibo_traffic(
                    row['retweeters_text'], weibo_traffic_set=weibo_traffic_dict_set), axis=1)]
                result_dataframe_list.append(dataframe_select)
            else:
                print('{} has been processed!'.format(file))

        if len(result_dataframe_list) > 6:
            print('Save file to local...')
            counter += 1
            concat_dataframe = pd.concat(result_dataframe_list)
            concat_dataframe.to_csv(os.path.join(save_path, 'shanghai_traffic_{}.csv'.format(counter)))
            result_dataframe_list = []

    concat_dataframe = pd.concat(result_dataframe_list)
    concat_dataframe.to_csv(os.path.join(save_path, 'shanghai_traffic_final.csv'.format(counter)))


def get_traffic_weibo(weibo_dataframe: pd.DataFrame, traffic_word_set: set) -> pd.DataFrame:
    """
    Return candidate traffic-related Weibo dataframe
    :param weibo_dataframe: a pandas dataframe containing Weibo data
    :param traffic_word_set: a traffic-related dictionary
    :return: a candidate traffic Weibo dataframe
    """
    decision1 = weibo_dataframe['text'].str.contains("|".join(traffic_word_set))
    decision2 = weibo_dataframe['retweeters_text'].str.contains("|".join(traffic_word_set))
    # We select rows of which the original post or repost contain at least one traffic-related word
    traffic_dataframe_post = weibo_dataframe[decision1 | decision2]
    traffic_dataframe_reset_index = traffic_dataframe_post.reset_index(drop=True)
    return traffic_dataframe_reset_index


def get_weibos_from_users_json(data_path, json_filename, save_path, user_set):
    """
    Get Weibo data posted from a user id set based on the json files
    :param data_path: path saving the Sina Weibo data
    :param json_filename: the studied json_filename
    :param save_path: path used to save the data
    :param user_set: a set containing the studied user ids
    :return: None
    """
    with open(os.path.join(data_path, json_filename), encoding='utf-8', errors='ignore') as json_file:

        time_list = []
        user_id_list = []
        weibo_id_list = []
        text_list = []
        geo_type_list = []
        lat_list = []
        lon_list = []
        retweeters_id_list = []
        retweets_weibo_id_list = []
        retweeters_text_list = []

        counter = 0

        for line in json_file:

            try:
                data = json.loads(line)
                tweet_list = data['statuses']
                for index, weibo in enumerate(tweet_list):
                    decision1 = weibo['uid'] in user_set
                    try:
                        decision2 = weibo['retweeted_status']['uid'] in user_set
                    except (KeyError, TypeError) as e:
                        decision2 = False

                    if decision1 or decision2:
                        time_list.append(weibo['created_at'])
                        user_id_list.append(weibo['uid'])
                        text_list.append(weibo['text'])
                        weibo_id_list.append(weibo['id'])
                        try:
                            weibo_geo_info = weibo['geo']
                            lat_list.append(str(weibo_geo_info['coordinates'][0]))
                            lon_list.append(str(weibo_geo_info['coordinates'][1]))
                            geo_type_list.append(weibo_geo_info['type'])
                        except (KeyError, TypeError) as e:
                            lat_list.append('Not Given')
                            lon_list.append('Not Given')
                            geo_type_list.append('Not Given')
                        try:
                            retweeted_statuses = weibo['retweeted_status']
                            retweeters_id_list.append(retweeted_statuses['uid'])
                            retweeters_text_list.append(retweeted_statuses['text'])
                            retweets_weibo_id_list.append(retweeted_statuses['id'])
                        except (KeyError, TypeError) as e:
                            retweeters_id_list.append(0)
                            retweeters_text_list.append('no retweeters')
                            retweets_weibo_id_list.append('no retweets')
            except JSONDecodeError as e_1:
                print('The error line is: {}'.format(line))
                print('ignore')
            except TypeError as e_2:
                print('The error line is: {}'.format(line))
                print('ignore')

            if len(weibo_id_list) > 100000:
                counter += 1

                print('{} weibos have been processed!'.format(len(weibo_id_list)))
                result_dataframe = pd.DataFrame(columns=['author_id', 'weibo_id', 'created_at', 'text', 'lat',
                                                         'lon', 'loc_type', 'retweeters_ids',
                                                         'retweeters_text', 'retweets_id'])
                result_dataframe['author_id'] = user_id_list
                result_dataframe['weibo_id'] = weibo_id_list
                result_dataframe['created_at'] = time_list
                result_dataframe['text'] = text_list
                result_dataframe['lat'] = lat_list
                result_dataframe['lon'] = lon_list
                result_dataframe['loc_type'] = geo_type_list
                result_dataframe['retweeters_ids'] = retweeters_id_list
                result_dataframe['retweeters_text'] = retweeters_text_list
                result_dataframe['retweets_id'] = retweets_weibo_id_list
                result_dataframe.to_csv(os.path.join(save_path, json_filename[:9] + '_{}.csv'.format(counter)),
                                        encoding='utf-8')
                # release memory
                user_id_list = []
                weibo_id_list = []
                time_list = []
                text_list = []
                lat_list = []
                lon_list = []
                geo_type_list = []
                retweeters_id_list = []
                retweeters_text_list = []
                retweets_weibo_id_list = []

        result_dataframe = pd.DataFrame(columns=['author_id', 'weibo_id', 'created_at', 'text', 'lat',
                                                 'lon', 'loc_type', 'retweeters_ids',
                                                 'retweeters_text', 'retweets_id'])
        result_dataframe['author_id'] = user_id_list
        result_dataframe['weibo_id'] = weibo_id_list
        result_dataframe['created_at'] = time_list
        result_dataframe['text'] = text_list
        result_dataframe['lat'] = lat_list
        result_dataframe['lon'] = lon_list
        result_dataframe['loc_type'] = geo_type_list
        result_dataframe['retweeters_ids'] = retweeters_id_list
        result_dataframe['retweeters_text'] = retweeters_text_list
        result_dataframe['retweets_id'] = retweets_weibo_id_list
        result_dataframe.to_csv(os.path.join(save_path, json_filename[:9] + '_final.csv'), encoding='utf-8')


def get_weibos_from_users_csv(path, filename, save_path, save_filename, user_set):
    with open(os.path.join(path, filename), 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        counter = 1
        with open(os.path.join(save_path, save_filename), 'w', encoding='utf-8') as new_file:
            field_names = ['author_id', 'weibo_id', 'created_at', 'text', 'lat', 'lon', 'retweeters_ids',
                           'retweeters_text', 'retweets_id']
            csv_writer = csv.DictWriter(new_file, fieldnames=field_names)
            csv_writer.writeheader()
            for line in csv_reader:
                decision1 = line['author_id'] in user_set
                decision2 = line['retweeters_ids'] in user_set
                if decision1 or decision2:
                    print('found {} weibo'.format(counter))
                    csv_writer.writerow(line)



def build_data_for_label(dataframe: pd.DataFrame, traffic_word_set: set, interested_months_list: list,
                         timezone):
    """
    Build the dataframe for manual label
    :param dataframe: a Weibo dataframe
    :param traffic_word_set: a traffic-related dictionary
    :param interested_months_list: a list which contains the interested months
    :param timezone: the studied timezone
    :return: a dataframe for manual label
    """
    # Check whether the column names are valid
    assert 'text' in dataframe, 'Check the dataframe!'
    assert 'retweeters_text' in dataframe, 'Check the dataframe!'

    # Use weibo in specific time range as the data for labeling
    dataframe['local_time'] = dataframe.apply(lambda row: transform_string_time_to_datetime(
        row['created_at'], target_time_zone=timezone, convert_utc_time=False), axis=1)
    dataframe_sorted = dataframe.sort_values(by='local_time')
    dataframe_sorted['month'] = dataframe_sorted.apply(lambda row: row['local_time'].month, axis=1)
    dataframe_select = dataframe_sorted.loc[dataframe_sorted['month'].isin(interested_months_list)]

    # Get the author and reposters' data
    author_dataframe = dataframe_select[['author_id', 'weibo_id', 'text']]
    reposter_dataframe = dataframe_select[['retweeters_ids', 'retweets_id', 'retweeters_text']]
    # get the dataframe that contains the reposts
    reposter_dataframe_without_nan = reposter_dataframe.loc[reposter_dataframe['retweeters_ids'] != '0']
    reposter_data_copy = reposter_dataframe_without_nan.copy()
    reposter_data_copy['retweeters_ids'] = reposter_data_copy.apply(lambda row: row['retweeters_ids'][1:-1], axis=1)
    reposter_data_copy['retweets_id'] = reposter_data_copy.apply(lambda row: row['retweets_id'][1:-1], axis=1)
    reposter_data_copy['retweeters_text'] = reposter_data_copy.apply(lambda row: row['retweeters_text'][1:-1], axis=1)

    # rename columns
    reposter_data_renamed = reposter_data_copy.rename(columns={'retweeters_ids': 'author_id',
                                                               'retweets_id': 'weibo_id',
                                                               'retweeters_text': 'text'})

    combined_dataframe = pd.concat([author_dataframe, reposter_data_renamed], axis=0)
    # drop the duplicate rows based on the Weibo id
    combined_dataframe_without_duplicates = combined_dataframe.drop_duplicates(subset=['weibo_id'], keep='first')
    combined_dataframe_without_na = combined_dataframe_without_duplicates[
        ~combined_dataframe_without_duplicates['text'].isna()]
    # Get the weibos containing at least one keyword
    combined_dataframe_candidate_traffic = combined_dataframe_without_na[
        combined_dataframe_without_na['text'].str.contains('|'.join(traffic_word_set))]
    combined_dataframe_final = combined_dataframe_candidate_traffic.reset_index(drop=True)
    return combined_dataframe_final


def get_weibo_from_users(path, user_set):
    """
    Get the Weibo data posted from a set of users
    :param path: a Weibo data path
    :param user_set: a set of social media users
    :return: a Weibo dataframe posted from a set of users
    """
    official_account_data_list = []
    for file in os.listdir(path):
        print('Coping with the file: {}'.format(file))
        dataframe = pd.read_csv(os.path.join(path, file), encoding='utf-8', index_col=0)
        decision1 = (dataframe['author_id'].isin(user_set))
        decision2 = (dataframe['retweeters_ids'].isin(user_set))
        dataframe_select = dataframe.loc[decision1 | decision2]
        official_account_data_list.append(dataframe_select)
    concat_data = pd.concat(official_account_data_list, axis=0)
    return concat_data


def get_traffic_weibo_and_reposts(dataframe: pd.DataFrame):
    """
    Get the Weibo data with traffic-related reposts
    :param dataframe: a Weibo dataframe
    :return: pandas dataframe, a Weibo id set
    This function would also print the number of unique Weibo ids, considering original post and repost
    """
    origin_posts = dataframe.loc[dataframe['traffic_prediction_retweets'].isin([1,2])]
    processed_retweet_id_str = list(origin_posts['retweets_id'])
    processed_retweet_id = [np.int64(id_str) for id_str in processed_retweet_id_str]
    assert 0 in processed_retweet_id, 'Some Weibos do not have retweets at all'
    all_processed_ids = set(list(origin_posts['weibo_id']) + processed_retweet_id)
    print('Number of processed ids: {}'.format(len(all_processed_ids)))
    origin_posts_reindex = origin_posts.reset_index(drop = True)
    return origin_posts_reindex, all_processed_ids


def get_traffic_weibo_no_retweets(dataframe, processed_ids) -> pd.DataFrame:
    """
    Get the Weibo data which is traffic-related but does not have retweets
    :param dataframe: a Weibo dataframe
    :param processed_ids: a set of Weibo ids which have been processed(processed by get_traffic_weibo_and_reposts func)
    :return: a pandas dataframe which contains traffic-related Weibo but does not have retweets
    """
    decision1 = (dataframe['traffic_prediction_text'].isin([1,2]))
    decision2 = (dataframe['traffic_prediction_retweets'] == 0)
    filtered_posts = dataframe.loc[decision1 & decision2]
    # make sure there is no overlapping between filtered_posts and processed_ids
    final_posts = filtered_posts.loc[~filtered_posts['weibo_id'].isin(processed_ids)]
    final_posts_reindex = final_posts.reset_index(drop = True)
    print('Number of weibos: {}'.format(final_posts.shape[0]))
    return final_posts_reindex


def count_id_posted_traffic(path):
    """
    Get the author & retweeter id who have posted traffic-related information
    :param path: a studied path
    :return: author id, retweeter id dataframe sorted by the number of times posted traffic information
    in descending order
    """
    traffic_data_list = []
    for file in os.listdir(path):
        print('Coping with the file: {}'.format(file))
        dataframe = pd.read_csv(os.path.join(path, file), encoding='utf-8', index_col=0)
        decision1 = (dataframe['traffic_prediction_text'].isin([1, 2]))
        decision2 = (dataframe['traffic_prediction_retweets'].isin([1, 2]))
        dataframe_select = dataframe.loc[decision1 | decision2]
        traffic_data_list.append(dataframe_select)
    concat_data_traffic = pd.concat(traffic_data_list, axis=0)

    author_dataframe = pd.DataFrame(columns=['author_id', 'count'])
    author_counter = Counter(concat_data_traffic['author_id'])
    author_dataframe['author_id'] = list(author_counter.keys())
    author_dataframe['count'] = [author_counter[key] for key in list(author_dataframe['author_id'])]
    author_sorted_dataframe = author_dataframe.sort_values(by='count', ascending=False).reset_index(drop=True)

    retweeter_dataframe = pd.DataFrame(columns=['retweeter_id', 'count'])
    retweeter_counter = Counter(concat_data_traffic['retweeters_ids'])
    retweeter_dataframe['retweeter_id'] = list(retweeter_counter.keys())
    retweeter_dataframe['count'] = [retweeter_counter[key] for key in list(retweeter_dataframe['retweeter_id'])]
    retweeter_sorted_dataframe = retweeter_dataframe.sort_values(by='count', ascending=False).reset_index(drop=True)

    return author_sorted_dataframe, retweeter_sorted_dataframe




if __name__ == '__main__':
    # sample_weibo = "【沪蓉高速四川广安汽车追尾事故 致11死1伤】今天13时30分许，在沪蓉高速公路广安市岳池县境内南充往广安" \
    #                "方向，一辆长安面包车与一辆停靠在路边修理轮胎的重型大货车追尾，造成11人死亡，1人重伤。" \
    #                "http://t.cn/zWm0qoM新华网"
    # sample_weibo_object = Chinese_Weibo_Analyze(sample_weibo)
    # nouns_list, verbs_list = sample_weibo_object.parse_chinese
    # print('Nouns: {}'.format(nouns_list))
    # print('Verbs: {}'.format(verbs_list))

    # traffic_data_official_account = get_weibo_from_users(path=data_paths.shanghai_dataframes_traffic_sentiment,
    #                                             user_set=traffic_acount_uid_final)
    # traffic_data_official_account.to_excel(os.path.join(data_paths.desktop, 'traffic_official_account.xlsx'))
    print('Load the data...')
    weibo_shanghai = pd.read_csv(os.path.join(data_paths.chinese_weibo_data_path, 'shanghai_weibo_apr_june.csv'),
                                 encoding='utf-8', index_col = 0)
    print('The data has been loaded!')
    data_for_label = build_data_for_label(dataframe=weibo_shanghai, traffic_word_set=traffic_word_set_update,
                                          interested_months_list=[4,5], timezone=timezone_shanghai)
    data_for_label.to_excel(os.path.join(data_paths.desktop, 'data_for_label.xlsx'))