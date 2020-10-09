import pandas as pd
import numpy as np
from collections import Counter
import time

from matplotlib import pyplot as plt
import seaborn as sns
import folium

import data_paths


def create_dist_from_dict(counter_dict, percentile_value: int = None, only_return_dataframe=True):

    """
    Create the historgram based on percentile value
    :param counter_dict: a python dictionary storing the number of times an account appear
    :param percentile_value: the predefined percentile value
    :param only_return_dataframe: whether we only return the dataframe or not
    :return: a histogram and the pandas dataframe saving the account id and corresponding number of appearance
    """

    dict_keys = list(counter_dict.keys())
    dict_values = [counter_dict[key] for key in dict_keys]

    count_dataframe = pd.DataFrame(columns=['account_id', 'count'])
    count_dataframe['account_id'] = dict_keys
    count_dataframe['count'] = dict_values
    count_dataframe_sorted = count_dataframe.sort_values(by='count', ascending=False)
    count_dataframe_reindex = count_dataframe_sorted.reset_index(drop=True)

    if percentile_value is not None:
        count_array = np.array(list(count_dataframe_reindex['count']))
        count_percentile_value = np.percentile(count_array, percentile_value)
        count_dataframe_select = count_dataframe_reindex.loc[count_dataframe_reindex['count'] <= count_percentile_value]
        plot_count_list = list(count_dataframe_select['count'])
        proportion_tweet = sum(plot_count_list) / sum(list(count_dataframe_reindex['count']))
        proportion_account = len(plot_count_list) / len(list(count_dataframe_reindex['count']))
        print('The {} percentile value is: {}'.format(percentile_value, count_percentile_value))
        print('The proportion of selected tweets is: {}'.format(proportion_tweet))
        print('The proportion of selected accounts is: {}'.format(proportion_account))
    else:
        print('We take all the accounts into account!')
        plot_count_list = list(count_dataframe_reindex['count'])

    if only_return_dataframe:
        print('Only return the dataframe')
        return count_dataframe_reindex
    else:
        print('Plot the histogram')
        fig, axis = plt.subplots(1, 1, figsize=(10, 8))
        sns.distplot(plot_count_list, ax=axis, color='b')
        axis.set_xlabel('Number of Posted Tweets')
        plt.show()
        return count_dataframe_reindex


def plot_tweet_count_cdf(account_count_dataframe, title_name, candidate_threshold=None):

    """
    Create a histogram based on the number times an account appears in the tweet dataset
    :param account_count_dataframe: a pandas dataframe saving the account id and number of appearance
    :param title_name: the title of the plot
    :param candidate_threshold: a test threshold to get the bot social media accounts
    """

    account_count_dataframe_copy = account_count_dataframe.copy()
    account_count_dataframe_copy['log'] = account_count_dataframe_copy.apply(lambda row: np.log(row['count']), axis=1)
    count_series = pd.Series(account_count_dataframe_copy['log'])

    fig, axis = plt.subplots(1, 1, figsize=(10, 8))
    count_series.hist(cumulative=True, density=1, bins=20, histtype='step', color='blue', grid=False, ax=axis)
    axis.set_title(title_name, size=12)
    if candidate_threshold is not None:
        axis.axvline(np.log(candidate_threshold), color='black')
        axis.text(np.log(candidate_threshold) + 0.3, 0.6, 'Number of posted tweets: \n{}'.format(candidate_threshold),
                  size=12)
    plt.show()


def create_folium_map(dataframe:pd.DataFrame, lat_lon_range:list, points_considered:int):
    """
    Create a folium interactive map for some random sampled tweets posted in one city
    :param dataframe: a tweet dataframe
    :param lat_lon_range: a latitude and longitude range
    :param points_considered: the number tweets plotted on the map
    :return: a folium map containing the random sampled tweets posted in one city
    """
    created_map = folium.Map(location=[dataframe['lat'].mean(), dataframe['lon'].mean()], zoom_start=8)
    dataframe['pos'] = dataframe.apply(lambda row: (row['lat'], row['lon']), axis=1)
    if dataframe.shape[0] > points_considered: # Check the dataframe size
        dataframe_sampled = dataframe.sample(points_considered)
    else:
        dataframe_sampled = dataframe.copy()
    # Draw the tweets on the interactive map
    for index, row in dataframe_sampled.iterrows():
        created_map.add_child(folium.CircleMarker(location=(row['pos'][0], row['pos'][1]), radius=1, color='black',
                                                  alpha=0.3))
    # Draw the pre-defined bounding box
    created_map.add_child(folium.Rectangle(bounds=[[[lat_lon_range[1], lat_lon_range[0]],
                                                    [lat_lon_range[3], lat_lon_range[2]]]]))
    created_map.add_child(folium.Marker(location=[lat_lon_range[1], lat_lon_range[0]]))
    created_map.add_child(folium.Marker(location=[lat_lon_range[3], lat_lon_range[2]]))
    return created_map