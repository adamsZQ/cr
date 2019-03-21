import json
import numpy as np

from sklearn.model_selection import train_test_split

from baseline.max_entropy import max_entropy, max_entropy_4all
from recommend.recommender import Recommender
from tools.sql_tool import select_by_attributes, select_genres


class PolicyBaseline():

    def __init__(self, recommender, file_path, max_turn=5):
        self.file_path = file_path
        self.__load_data()
        self.question_sequence = max_entropy_4all(file_path)
        self.max_turn = max_turn
        self.recommender = recommender

    def __load_data(self):
        data_list = []
        with open(self.file_path, 'r') as f:
            for line in f:
                line = json.loads(line)
                data_list.append(line)
        trainset, testset, a, b = train_test_split(data_list, [0] * len(data_list), test_size=0.2, random_state=1)
        self.data = testset

    def simulate(self, top_k=5, match_all_genres=True):
        attributes = {}
        correct_num = 0
        for line in self.data:
            print('---------------------------')
            for i in range(self.max_turn):
                attributes[self.question_sequence[i]] = line[self.question_sequence[i]]
            print(attributes)
            target = line['movie']

            # pop genres to check separately
            no_genres = True
            if 'genres' in attributes:
                no_genres = False
                target_genres = (attributes.pop('genres')).split('|')
                target_genres = [int(genre) for genre in target_genres]
            # get id from database those match all attributes without genres
            id_list_nogenre = select_by_attributes(attributes)

            if no_genres:
                # if genres doesn't in attribute, skip checking genre
                id_list_match_genre = id_list_nogenre
            # check genres
            else:
                id_list_match_genre = []
                for id in id_list_nogenre:
                    genre_list = select_genres(id)
                    # print('+++++++++=', target_genres, 'sadfsdaf', genre_list)
                    if match_all_genres:
                        # target_genres must be subset of genre list
                        if set(target_genres).issubset(set(genre_list)):
                            id_list_match_genre.append(id)
                            print('add', id)
                    else:
                        # check if they have at least one element common
                        if set(target_genres) & set(genre_list):
                            id_list_match_genre.append(id)
                            print('add', id)
            # predict rating of movie matching all attributes
            item_sort, predict = self.recommender.predict(line['user'], id_list_match_genre)

            index_upsort = np.argsort(predict)
            index_downsort = index_upsort[::-1]

            top_k_items = [id_list_match_genre[index] for index in index_downsort[:top_k]]
            
            print('result', list(zip(item_sort, predict)))

            if int(target) in top_k_items:
                print('succeed!target:{}, top5:{}'.format(target, top_k_items ))
                correct_num = correct_num + 1
            else:
                print('fail!target:{}, top5:{}'.format(target, top_k_items))

        accuracy = float(correct_num)/float(len(self.data))
        print('finish!, accuracy is:', accuracy)


if __name__ == '__main__':
    recommender = Recommender('/path/mv/', 'model/knn_model.m', 'ratings_cleaned.dat')
    policy = PolicyBaseline(recommender, '/path/mv/movie_rating', max_turn=1)
    policy.simulate(top_k=1, match_all_genres=True)