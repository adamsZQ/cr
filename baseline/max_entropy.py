import json
from collections import Counter

import numpy as np


def increment(data, dict, match_all_genres=False):
    if match_all_genres:
        if len(dict) == 0:
            dict[data] = 1
        else:
            for key in list(dict.keys()):
                key_set = set(key.split('|'))
                data_set = set(data.split('|'))

                if key_set == data_set:
                    dict[data] = dict[data] + 1
                else:
                    dict[data] = 1
        return dict
    else:
        if data not in dict.keys():
            dict[data] = 1
        else:
            dict[data] = dict[data] + 1
        return dict


def entropy(index_list, num_list):
    ent = 0.0

    sum_num = sum(num_list)
    for index,num in zip(index_list, num_list):
        pi = num / sum_num
        logp = np.log2(pi)
        ent -= pi * logp

    return ent


def load_data(file_path):
    rating_movie_list = []
    with open(file_path) as f:
        for line in f:
            line = json.loads(line)
            line['user'] = int(line['user'])
            line['movie'] = int(line['movie'])
            line['rating'] = float(line['rating'])
            line['critic_rating'] = int(line['critic_rating'])
            line['audience_rating'] = int(line['audience_rating'])
            line['director'] = int(line['director'])
            line['country'] = int(line['country'])
            genres = []
            line['genres_str'] = line['genres']
            for genre in line['genres'].split('|'):
                genres.append(int(genre))
            line['genres'] = genres

            rating_movie_list.append(line)
    return rating_movie_list


def max_entropy_4all(file_path, match_all_genres=False):
    rating_movie_list = load_data(file_path)
    critic_rating = {}
    audience_rating = {}
    director = {}
    country = {}
    genres = {}

    for data in rating_movie_list:

        critic_rating = increment(data['critic_rating'], critic_rating)
        audience_rating = increment(data['audience_rating'], audience_rating)
        director = increment(data['director'], director)
        country = increment(data['country'], country)

        if match_all_genres:
            genres = increment(data['genres_str'], genres, match_all_genres)
        else:
            for genre in data['genres']:
                genres = increment(genre, genres)

    ent_index = {0: 'critic_rating', 1: 'audience_rating', 2: 'director', 3: 'country', 4: 'genres'}

    ent_critic = entropy(critic_rating.keys(), critic_rating.values())
    ent_audience = entropy(audience_rating.keys(), audience_rating.values())
    ent_director = entropy(director.keys(), director.values())
    ent_country = entropy(country.keys(), country.values())
    ent_genres = entropy(genres.keys(), genres.values())

    ent = [ent_critic, ent_audience, ent_director, ent_country, ent_genres]
    print(ent)

    ent_sort = np.argsort(ent)
    ent_sort = ent_sort[::-1]
    print([ent_index[index] for index in ent_sort])
    return [ent_index[index] for index in ent_sort]


def max_entropy(file_path, match_all_genres=False):
    rating_movie_list = load_data(file_path)

    critic_rating = {}
    audience_rating = {}
    director = {}
    country = {}
    genres = {}

    user_id = rating_movie_list[0]['user']
    index_list_1 = []
    index_list_2 = []
    index_list_3 = []
    index_list_4 = []
    index_list_5 = []

    for data in rating_movie_list:
        if data['user'] == user_id:
            critic_rating = increment(data['critic_rating'], critic_rating)
            audience_rating = increment(data['audience_rating'], audience_rating)
            director = increment(data['director'], director)
            country = increment(data['country'], country)

            if match_all_genres:
                genres = increment(data['genres_str'], genres, match_all_genres)
            else:
                for genre in data['genres']:
                    genres = increment(genre, genres)
        else:
            user_id = data['user']
            # print('-------------------------------------------------------------')
            ent_critic = entropy(critic_rating.keys(), critic_rating.values())
            ent_audience = entropy(audience_rating.keys(), audience_rating.values())
            ent_director = entropy(director.keys(), director.values())
            ent_country = entropy(country.keys(), country.values())
            ent_genres = entropy(genres.keys(), genres.values())

            user_ent = [ent_critic, ent_audience, ent_director, ent_country, ent_genres]
            ent_index = {0:'critic_rating', 1:'audience_rating', 2:'director', 3:'country', 4:'genres'}
            index_list_1.append(user_ent.index(max(user_ent)))
            user_ent[user_ent.index(max(user_ent))] = -1
            index_list_2.append(user_ent.index(max(user_ent)))
            user_ent[user_ent.index(max(user_ent))] = -1
            index_list_3.append(user_ent.index(max(user_ent)))
            user_ent[user_ent.index(max(user_ent))] = -1
            index_list_4.append(user_ent.index(max(user_ent)))
            user_ent[user_ent.index(max(user_ent))] = -1
            index_list_5.append(user_ent.index(max(user_ent)))
            user_ent[user_ent.index(max(user_ent))] = -1

            critic_rating = {}
            audience_rating = {}
            director = {}
            country = {}
            genres = {}

            critic_rating = increment(data['critic_rating'], critic_rating)
            audience_rating = increment(data['audience_rating'], audience_rating)
            director = increment(data['director'], director)
            country = increment(data['country'], country)
            if match_all_genres:
                genres = increment(data['genres_str'], genres, match_all_genres)
            else:
                for genre in data['genres']:
                    genres = increment(genre, genres)

    # get max_extropy
    print(Counter(index_list_1))
    print(Counter(index_list_2))
    print(Counter(index_list_3))
    print(Counter(index_list_4))
    print(Counter(index_list_5))

    aa = ent_index[list(Counter(index_list_1))[0]]
    bb = ent_index[list(Counter(index_list_2))[0]]
    cc = ent_index[list(Counter(index_list_3))[0]]
    dd = ent_index[list(Counter(index_list_4))[0]]
    ee = ent_index[list(Counter(index_list_5))[0]]

    return [aa, bb, cc, dd, ee]


if __name__ == '__main__':
    aa = max_entropy_4all('/path/mv/movie_rating', match_all_genres=False)
    # print(aa)