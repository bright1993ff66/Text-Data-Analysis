from datetime import datetime
import os
import pytz


def create_pytz_timezone(time_zone_string):
    """Create a time zone based on a datetime object timezone string.
    For a list of timezone strings offered by pytz, please check
    https://stackoverflow.com/questions/13866926/is-there-a-list-of-pytz-timezones"""
    result_timezone = pytz.timezone(time_zone_string)
    return result_timezone


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


def transform_datetime_string_to_datetime(time_string, target_time_zone):
    """Transform the datetime string object to datetime object"""
    datetime_object = datetime.strptime(time_string, '%Y-%m-%d %H:%M:%S%z')
    final_time_object = datetime_object.replace(tzinfo=target_time_zone)
    return final_time_object


def weekday_hour_attributes(dataframe):
    """Get the weekday and hour attributes of weibos"""
    dataframe_copy = dataframe.copy()
    dataframe_copy['weekday'] = dataframe_copy.apply(lambda row: row['local_time'].weekday(), axis = 1)
    dataframe_copy['hour'] = dataframe_copy.apply(lambda row: row['local_time'].hour, axis = 1)
    return dataframe_copy


def get_tweets_in_some_years(dataframe, target_time_zone, year):

    """Get the tweets posted in some years"""

    assert 'local_time' in dataframe, 'Please add the local_time column to the dataframe'

    dataframe['local_time'] = dataframe.apply(lambda row: transform_string_time_to_datetime(row['created_at'],
                                                                                            target_time_zone), axis=1)
    dataframe['year'] = dataframe.apply(lambda row: row['local_time'].year, axis=1)
    if type(year) == list:
        print('The years we consider are: {}'.format(year))
        dataframe_select = dataframe.loc[dataframe['year'].isin(year)]
    elif type(year) == int:
        print('We only consider one year, which is: {}'.format(year))
        dataframe_select = dataframe.loc[dataframe['year'] == year]
    else:
        print('The type of the argument year is not correct! It should be either list or int')
        raise ValueError('The type of the argument year is not right!')
    dataframe_select_sorted = dataframe_select.sort_values(by='local_time')
    dataframe_select_reindex = dataframe_select_sorted.reset_index(drop=True)

    return dataframe_select_reindex


if __name__ == '__main__':
    london_time_zone = create_pytz_timezone('Europe/London')
    hk_time_zone = create_pytz_timezone('Asia/Shanghai')
    sample_london_time = 'Sun Dec 31 23:00:17 +0000 2017'
    sample_london_datetime = transform_string_time_to_datetime(time_string=sample_london_time,
                                                               target_time_zone=london_time_zone)
    print(sample_london_datetime)
