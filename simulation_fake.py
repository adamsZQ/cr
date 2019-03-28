import argparse
import json
import time
import profile
import numpy as np

import torch
from sklearn.model_selection import train_test_split
from torch import optim

from recommend.recommender import Recommender
from tools.sql_tool import select_by_attributes, select_genres, select_all_movie_genres, select_all

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

# get part of datalist
data_list, useless, a, b = train_test_split(data_list, [0] * len(data_list), test_size=0.999, random_state=1)
# data_list, useless, a, b = train_test_split(data_list, [0] * len(data_list), test_size=0.9, random_state=1)
# train test split
trainset, testset, a, b = train_test_split(data_list, [0] * len(data_list), test_size=0.2, random_state=1)
print(len(trainset))
testset, valset, a, b = train_test_split(testset, [0] * len(testset), test_size=0.5, random_state=2)


actions = ['director', 'genres', 'critic_rating', 'country', 'audience_rating', 'recommendation']

movie_genres = select_all_movie_genres()

# print(movie_genres)

'''
    get all id-genres
'''
movie_id_list = []
for movie_genre in movie_genres:
    if movie_genre[0] not in movie_id_list:
        movie_id_list.append(movie_genre[0])

data_list = select_all()

id_genres_list = [None] * (max(movie_id_list) + 1)

for data in data_list:
    movie_id = data[0]
    genres = data[4].split('|')

    genres = [int(genre) for genre in genres]
    id_genres_list[movie_id] = genres


def get_genres(movie_id):
    return id_genres_list[movie_id]


def simulate(model, recommender, max_dialength=7, max_recreward=50, r_c=-1, r_q=-10):
    print('simulate start')
    num_epochs = 10000

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    for epoch in range(num_epochs):
        reward_list = []
        conversation_turn_num = []
        correct_num = 0
        quit_num = 0
        t_start = time.time()
        t_rec = 0
        for data in trainset:
            director = data['director']
            genres = genres_list.index(set(data['genres'].split('|')))
            critic_rating = data['critic_rating']
            country = data['country']
            audience_rating = data['audience_rating']

            target = data['movie']

            data = [director, genres, critic_rating, country, audience_rating]
            state = [-1] * len(data)

            reward = 0

            for i in range(max_dialength):
                # select action
                # print('----------------------------------', i)
                action = model.select_action(np.array(state),device)
                # print('state', state)
                #
                # print('action', action)

                # if action asks question
                if action in range(5):
                    # if max_dialog length still asking question, give r_q
                    if i == max_dialength - 1:
                        # print('over length')
                        reward = r_q
                        quit_num = quit_num + 1
                        break
                    else:
                        # print('ask question')
                        state[action] = data[action]
                        reward = r_c
                # if action is recommendation
                elif action == 5:
                    # reward = max_recreward
                    t_rec_start = time.time()
                    if recommendation(state, target, recommender):
                        # recommend successfully
                        # print('recommend success')
                        reward = max_recreward
                        correct_num = correct_num + 1
                    else:
                        # fail
                        # print('recommend fail')
                        # TODO remove fail recommend negative reward
                        # reward = r_q
                        reward = 0
                    break
                else:
                    # print('wrong action')
                    model.rewards.append(r_q)
                    break

                # append reward
                #print('reward',reward)
                model.rewards.append(reward)
                reward_list.append(reward)

            # append reward
            #print('reward', reward)

            model.rewards.append(reward)
            reward_list.append(reward)
            # append conversation turn num
            conversation_turn_num.append(i+1)
            # update policy
            #print('update')
            model.update_policy(optimizer)

        if epoch % 1 == 0:
            print('sequence time:', time.time()-t_start)
            print('rec time:', t_rec)

            train_ave_reward = np.mean(reward_list)
            # ave_reward = np.mean(reward_list)
            ave_conv = np.mean(conversation_turn_num)
            accuracy = float(correct_num) / len(trainset)
            quit_rating = float(quit_num) / len(trainset)


            #val_ave_reward, val_ave_conv, val_accuracy = val(model, recommender, max_dialength, max_recreward,device, r_c, r_q)

            # ave_reward, ave_conv, accuracy = val(model, recommender, max_dialength, max_recreward, r_c, r_q)
            print('Epoch[{}/{}]'.format(epoch, num_epochs) +
                  'train ave_reward: {:.6f}'.format(train_ave_reward) +
                  'accuracy_score: {:.6f}'.format(accuracy) +
                  'ave_conversation: {:.6f}'.format(ave_conv) +
                  'quit_rating: {:.6f}'.format(quit_rating)
                  )
            # print('val_ave_reward: {:.6f}'.format(val_ave_reward) +
            #       'val_accuracy_score: {:.6f}'.format(val_accuracy) +
            #       'val_ave_conversation: {:.6f}'.format(val_ave_conv))


def val(model, recommender, max_dialength, max_recreward, device, r_c, r_q):
    reward_list = []
    conversation_turn_num = []
    correct_num = 0
    for data in valset:
        director = data['director']
        genres = genres_list.index(set(data['genres'].split('|')))
        critic_rating = data['critic_rating']
        country = data['country']
        audience_rating = data['audience_rating']

        target = data['movie']

        data = [director, genres, critic_rating, country, audience_rating]
        state = [-1] * len(data)

        reward = 0
        for i in range(max_dialength):
            # select action
            action = model.select_best_action(np.array(state), device)
            #print('-------------------------------action', action)
            # if action asks question
            if action in range(5):
                # if max_dialog length still asking question, give r_q
                if i == max_dialength:
                    reward = r_q
                    break
                else:
                    state[action] = data[action]
                    reward = r_c
            # if action is recommendation
            elif action == 5:
                if recommendation(state, target, recommender):
                    # recommend successfully
                    reward = max_recreward
                    correct_num = correct_num + 1
                else:
                    # fail
                    reward = r_q
                break
            else:
                model.rewards.append(r_q)
                break

            # append reward
            #print('reward',reward)
            model.rewards.append(reward)
            reward_list.append(reward)

        #print('reward', reward)
        model.rewards.append(reward)
        reward_list.append(reward)

        # append conversation turn num
        conversation_turn_num.append(i + 1)

    ave_reward = np.mean(reward_list)
    ave_conv = np.mean(conversation_turn_num)
    accuracy = float(correct_num) / len(data_list)

    return ave_reward, ave_conv, accuracy


def recommendation(states, target, recommender, top_k=5):
    index = [i for i in range(len(states)) if states[i] == -1]
    len_index = len(index)
    # print(len_index)
    # if len_index == 0:
    #     rating = 0.997187
    # elif len_index == 1:
    #     rating = 0.995071
    # elif len_index == 2:
    #     rating = 0.993133
    # elif len_index == 3:
    #     rating = 0.972280
    # elif len_index == 4:
    #     rating = 0.770491
    # elif len_index == 5:
    #     rating = 0.0

    if len_index == 0:
        rating = 0.889718
    elif len_index == 1:
        rating = 0.859617
    elif len_index == 2:
        rating = 0.836962
    elif len_index == 3:
        rating = 0.755233
    elif len_index == 4:
        rating = 0.341166
    elif len_index == 5:
        rating = 0.0

    success = np.random.choice(2,1,p=[1-rating,rating])

    if success == 1:
        return True
    else:
        return False


if __name__ == '__main__':
    if torch.cuda.is_available():
        print('using cuda')
        device = torch.device('cuda')
    else:
        print('using cpu')
        device = torch.device('cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix") # data and model prefix
    parser.add_argument("--file_name") # choose model(lstm/bilstm)
    parser.add_argument("--boundary_tags") # add START END tag
    args = parser.parse_args()

    FILE_PREFIX = args.prefix
    file_name = args.file_name

    if FILE_PREFIX is None:
        FILE_PREFIX ='/path/mv/'
    if file_name is None:
        file_name = 'model/5turns/policy_pretrain_1.2970.pkl'

    # file_name = '5turns/policy_pretrain_1.5979.pkl'
    model = torch.load(FILE_PREFIX+file_name).to(device)

    # recommender = Recommender(FILE_PREFIX, 'model/knn_model.m', 'ratings_cleaned.dat')
    recommender = None
    simulate(model, recommender,r_q=-40,r_c=0.5,max_recreward=10)