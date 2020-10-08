import os
import pandas as pd
import numpy as np
from collections import Counter
import networkx as nx
import tensorflow.keras.backend as K
import re
from random import sample
import pytz
from datetime import datetime

import data_paths


def delete_user(text):
    """
    Delete the @user in the weibo or tweet text
    :param text: a weibo or tweet string
    :return: a text string without @user
    """
    result_text = re.sub("@[^，，：：\s@:]+", "", text)
    return result_text


def number_of_tweet_users(dataframe, user_id_column_name, print_value=True):
    """
    Get the number of tweets and number of social media users
    :param dataframe: the studied dataframe
    :param user_id_column_name: the column name which saves the user ids
    :param print_value: whether print the values or not
    :return: the number of tweets and number of users
    """
    number_of_tweets = dataframe.shape
    number_of_users = len(set(dataframe[user_id_column_name]))
    # print('The column names are: {}'.format(dataframe.columns))
    if print_value:
        print('The number of tweets: {}; The number of unique social media users: {}'.format(
            number_of_tweets, number_of_users))
    else:
        return number_of_tweets, number_of_users


def read_local_file(path, filename, csv_file=True):
    """
    Read a csv or pickle file from a local directory
    :param path: the path which save the local csv file
    :param filename: the studied filename
    :param csv_file: whether this file is a csv file
    :return: a pandas dataframe
    """
    if csv_file:
        dataframe = pd.read_csv(os.path.join(path, filename), encoding='utf-8', index_col=0)
    else:
        dataframe = pd.read_pickle(os.path.join(path, filename))
    return dataframe


def get_edge_embedding_for_mlp(edge_list, embedding_dict, concatenate_or_not=True):
    """
    Get edge embeddings for mlp classifier
    :param edge_list: a python list which contains the edges of a graph
    :param embedding_dict: a dictionary of which key is the node and value is the node2vec embedding
    :param concatenate_or_not: whether we concatenate two node embeddings or not
    :return: the embeddings for edges of a graph
    """
    embs = []
    for edge in edge_list:
        node_id1 = edge[0]
        node_id2 = edge[1]
        emb1 = embedding_dict[node_id1]
        emb2 = embedding_dict[node_id2]
        if concatenate_or_not:
            emb_concat = np.concatenate([emb1, emb2], axis=0)
            embs.append(emb_concat)
        else:
            emb_multiply = np.multiply(emb1, emb2)
            embs.append(emb_multiply)
    return embs


def get_f1(y_true, y_pred):
    """
    Compute the F1 score based on truth and prediction
    :param y_true: the label
    :param y_pred: the prediction
    :return: return the F1 score for each batch data or for each epoch
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


def get_network_statistics(g):
    """Get the network statistics of a networkx graph"""
    num_connected_components = nx.number_connected_components(g)
    node_attribute_dict = nx.get_node_attributes(g, 'type')
    edge_attribute_dict = nx.get_edge_attributes(g, 'relation')
    user_dict = {key: value for (key, value) in node_attribute_dict.items() if value == 'user'}
    location_dict = {key: value for (key, value) in node_attribute_dict.items() if value == 'location'}
    user_user_edge = {key: value for (key, value) in edge_attribute_dict.items() if value == 'user_user'}
    location_user_edge = {key: value for (key, value) in edge_attribute_dict.items() if value == 'user_location'}
    num_nodes = nx.number_of_nodes(g)
    num_edges = nx.number_of_edges(g)
    density = nx.density(g)
    avg_clustering_coef = nx.average_clustering(g)
    avg_degree = sum([int(degree[1]) for degree in g.degree()]) / float(num_nodes)
    transitivity = nx.transitivity(g)

    if num_connected_components == 1:
        diameter = nx.diameter(g)
    else:
        diameter = None  # infinite path length between connected components

    network_statistics = {
        'num_connected_components': num_connected_components,
        'num_nodes': num_nodes,
        'num_user_nodes': len(user_dict),
        'num_location_nodes': len(location_dict),
        'num_edges': num_edges,
        'num_user_user_edges': len(user_user_edge),
        'num_user_location_edges': len(location_user_edge),
        'density': density,
        'diameter': diameter,
        'avg_clustering_coef': avg_clustering_coef,
        'avg_degree': avg_degree,
        'transitivity': transitivity
    }

    return network_statistics


def create_labelled_dataframe(dataframe):
    """
    Construct the labeled dataframe
    :param dataframe: a dataframe containing the labeled weibo dataframe
    :return: a final dataframe containing the weibo and reposted weibos
    """
    author_dataframe = dataframe[['weibo_id', 'text', 'label_1']]
    retweeter_dataframe = dataframe[['retweets_id', 'retweeters_text', 'label_2']]

    # Cope with the retweeter dataframe
    retweeter_dataframe_select = retweeter_dataframe.loc[retweeter_dataframe['label_2'] != -1]
    retweeter_dataframe_without_na = retweeter_dataframe_select[~retweeter_dataframe_select['label_2'].isna()]
    retweeter_data_without_duplicates = retweeter_dataframe_without_na.drop_duplicates(subset='retweets_id',
                                                                                       keep='first')
    retweeter_data_without_duplicates['retweets_id'] = retweeter_data_without_duplicates.apply(
        lambda row: row['retweets_id'][1:-1], axis=1)

    final_dataframe = pd.DataFrame(columns=['id', 'text', 'label'])
    author_id_list = list(author_dataframe['weibo_id'])
    author_text_list = list(author_dataframe['text'])
    author_label_list = list(author_dataframe['label_1'])
    author_id_list.extend(list(retweeter_data_without_duplicates['retweets_id']))
    author_text_list.extend(list(retweeter_data_without_duplicates['retweeters_text']))
    author_label_list.extend(list(retweeter_data_without_duplicates['label_2']))
    final_dataframe['id'] = author_id_list
    final_dataframe['text'] = author_text_list
    final_dataframe['label'] = author_label_list

    return final_dataframe


def combine_some_data(path, sample_num: int) -> pd.DataFrame:
    """Combine some random sampled dataframes from a local path"""
    files = os.listdir(path)
    random_sampled_files = sample(files, k=sample_num)
    dataframe_list = []

    for file in random_sampled_files:
        print('Coping with the file: {}'.format(file))
        dataframe = pd.read_csv(os.path.join(path, file), encoding='utf-8', index_col=0)
        dataframe_list.append(dataframe)

    concat_dataframe = pd.concat(dataframe_list, axis=0)
    concat_dataframe_reindex = concat_dataframe.reset_index(drop=True)
    return concat_dataframe_reindex


def transform_string_time_to_datetime(time_string, target_time_zone, convert_utc_time=True):
    """
    Transform the string time to the datetime
    :param time_string: a time string
    :param target_time_zone: the target time zone
    :param convert_utc_time: whether transfer the datetime object to utc first
    :return:
    """
    datetime_object = datetime.strptime(time_string, '%a %b %d %H:%M:%S %z %Y')
    if convert_utc_time:
        final_time_object = datetime_object.replace(tzinfo=pytz.utc).astimezone(target_time_zone)
    else:
        final_time_object = datetime_object.astimezone(target_time_zone)
    return final_time_object


def combine_candidate_ids(dataframe: pd.DataFrame) -> set:
    """
    Get the Weibo id set, considering original post and repost
    :param dataframe: a Weibo dataframe
    :return: a Weibo id set
    """
    author_int_set = set(dataframe['weibo_id'])
    retweeter_list = list(dataframe['retweets_id'])
    retweeter_int_set = set([np.int64(str(retweet_id[1:-1])) for retweet_id in retweeter_list if retweet_id != "['no retweets']"])
    # combine the retweet id and author id together
    combine_set = {*author_int_set, *retweeter_int_set}
    return combine_set


if __name__ == '__main__':
    graph = nx.read_gexf(os.path.join(data_paths.data_path, 'graph_structure', 'user_location_graph.gexf'))
    print(get_network_statistics(graph))
