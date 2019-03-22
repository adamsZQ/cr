import argparse
import json
import numpy as np

import torch
from sklearn.model_selection import train_test_split
from torch import optim

from recommend.recommender import Recommender
from tools.sql_tool import select_by_attributes, select_genres

data_list = []
director_list = []
genres_list = []
critic_rating_list = []
country_list = []
audience_rating_list = []
with open('/users4/chzhu/path/mv/movie_rating', 'r') as f:
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
# train test split
trainset, testset, a, b = train_test_split(data_list, [0] * len(data_list), test_size=0.2, random_state=1)
print(len(trainset))
testset, valset, a, b = train_test_split(testset, [0] * len(testset), test_size=0.5, random_state=2)


actions = ['director', 'genres', 'critic_rating', 'country', 'audience_rating', 'recommendation']


def simulate(model, recommender, max_dialength=7, max_recreward=50, r_c=-1, r_q=-10):
    print('simulate start')
    num_epochs = 10000

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    for epoch in range(num_epochs):
        reward_list = []
        conversation_turn_num = []
        correct_num = 0
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
                        break
                    else:
                        # print('ask question')
                        state[action] = data[action]
                        reward = r_c
                # if action is recommendation
                elif action == 5:
                    # reward = max_recreward
                    if recommendation(state, target, recommender):
                        # recommend successfully
                        # print('recommend success')
                        reward = max_recreward
                        correct_num = correct_num + 1
                    else:
                        # fail
                        # print('recommend fail')
                        reward = r_q
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
            print('evaluation')
            train_ave_reward = np.mean(reward_list)
            # ave_reward = np.mean(reward_list)
            ave_conv = np.mean(conversation_turn_num)
            accuracy = float(correct_num) / len(trainset)

            # ave_reward, ave_conv, accuracy = val(model, recommender, max_dialength, max_recreward, r_c, r_q)
            print('Epoch[{}/{}]'.format(epoch, num_epochs) + 'train ave_reward: {:.6f}'.format(train_ave_reward) +
                  'accuracy_score: {:.6f}'.format(accuracy) +
                  'ave_conversation: {:.6f}'.format(ave_conv))


        # 'val_ave_reward: {:.6f}'.format(ave_reward) +


def val(model, recommender, max_dialength, max_recreward, r_c, r_q):
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
            action = model.select_best_action(np.array(state))
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
        conversation_turn_num.append(i)

    ave_reward = np.mean(reward_list)
    ave_conv = np.mean(conversation_turn_num)
    accuracy = float(correct_num) / len(data_list)

    return ave_reward, ave_conv, accuracy


def recommendation(states, target, recommender, top_k = 5):
    target = target
    attributes = {}
    i = 0
    for state in states:
        if state != -1:
            attribute = actions[i]
            if attribute is 'genres':
                attributes[attribute] = genres_list[state]
            else:
                attributes[attribute] = state
        i = i + 1
    # print(attributes)
    # if genres is null
    if states[1] == -1:
        # pop genres to check separately
        no_genres = True
    else:
        no_genres = False
        try:
            target_genres = (attributes.pop('genres'))
            target_genres = [int(genre) for genre in target_genres]
        except Exception as e:
            print('states', states, 'attributes', attributes)
    # get id from database those match all attributes without genres
    id_list_nogenre = select_by_attributes(attributes)

    if no_genres:
        #print('nogenres', attributes)
        # if genres doesn't in attribute, skip checking genre
        id_list_match_genre = id_list_nogenre
    # check genres
    else:
        id_list_match_genre = []
        for id in id_list_nogenre:
            genre_list = select_genres(id)
            # print('+++++++++=', target_genres, 'sadfsdaf', genre_list)
            # target_genres must be subset of genre list
            if set(target_genres).issubset(set(genre_list)):
                id_list_match_genre.append(id)
                #print('add', id)
            else:
                pass
                #print(id,'does not match')
    # predict rating of movie matching all attributes
    item_sort, predict = recommender.predict(line['user'], id_list_match_genre)

    index_upsort = np.argsort(predict)
    index_downsort = index_upsort[::-1]

    top_k_items = [id_list_match_genre[index] for index in index_downsort[:top_k]]

    # print('result', list(zip(item_sort, predict)))

    if int(target) in top_k_items:
        #print('succeed!')
        return True
    else:
        #print('fail!')
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
    # file_name = '5turns/policy_pretrain_1.5979.pkl'
    model = torch.load(FILE_PREFIX+file_name).to(device)

    recommender = Recommender(FILE_PREFIX, 'model/knn_model.m', 'ratings_cleaned.dat')
    simulate(model, recommender, r_c=-1,max_recreward=50)