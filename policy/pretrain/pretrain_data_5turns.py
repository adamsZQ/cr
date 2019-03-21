import json
import random

import h5py as h5py
import numpy as np

from baseline.max_entropy import max_entropy_4all



data_list = []
director_list = []
genres_list = []
critic_rating_list = []
country_list = []
audience_rating_list = []
with open('/path/mv/movie_rating', 'r') as f:
    for line in f:
        line = json.loads(line)
        data_list.append(line)
        director_list.append(line['director'])
        genres_list.append(set(line['genres'].split('|')))
        critic_rating_list.append(line['critic_rating'])
        country_list.append(line['country'])
        audience_rating_list.append(line['audience_rating'])


def rule_based_action(brunch_num):
    # construct each 2000 data based on max entropy
    state_list = []
    action_list = []
    # TODO before supply, remove static question_sequence
    question_sequence = ['director', 'genres', 'critic_rating', 'country', 'audience_rating']
    # question_sequence = max_entropy_4all('/path/mv/movie_rating', match_all_genres=False)

    '''    
    actions = ['director', 'genres', 'critic_rating', 'country', 'audience_rating', 'recommendation]
    '''

    question_maxlen = len(question_sequence)

    # # first, construct 2000 start node data
    # state = [-1] * question_maxlen * (brunch_num * 2)
    # state_list.append(state)
    # # ask 0th question_sequence
    # action_list.append([0] * brunch_num)
    # # or recommend
    # action_list.append([5] * brunch_num)

    data_slice = random.sample(data_list, 2000)
    for data in data_slice:
        director = data['director']
        genres = genres_list.index(set(data['genres'].split('|')))
        critic_rating = data['critic_rating']
        country = data['country']
        audience_rating = data['audience_rating']

        state_init = [-1] * question_maxlen
        state_list.append(state_init.copy())
        action_list.append(0)

        state_init[0] = int(director)
        state_list.append(state_init.copy())
        action_list.append(1)

        state_init[1] = int(genres)
        state_list.append(state_init.copy())
        action_list.append(2)

        state_init[2] = int(critic_rating)
        state_list.append(state_init.copy())
        action_list.append(3)

        state_init[3] = int(country)
        state_list.append(state_init.copy())
        action_list.append(4)

        state_init[4] = int(audience_rating)
        state_list.append(state_init.copy())
        action_list.append(5)

        print('------------------------------')
        print(len(state_list))
        print(len(action_list))


    state_list = np.array(state_list).reshape(-1,question_maxlen)
    action_list = np.array(action_list).reshape(-1, 1)

    f = h5py.File('/path/mv/pretrain_data_5turns.h5', 'w')
    f.create_dataset(data=state_list, name='states')
    f.create_dataset(data=action_list, name='actions')

if __name__ == '__main__':
    rule_based_action(2000)