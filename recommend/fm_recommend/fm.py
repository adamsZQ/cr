import json

import numpy as np
import os
from fastFM import als
from sklearn.feature_extraction import DictVectorizer
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


class FM:
    def __init__(self,model_file = "", n_iter=1000, init_stdev=0.1, rank=2, l2_reg_w=0.1, l2_reg_V=0.5):
        if os.path.exists(model_file):
            print('old')
            self.fm = joblib.load(model_file)
        else:
            self.fm = als.FMRegression(n_iter, init_stdev, rank, l2_reg_w, l2_reg_V)
            print('new')

    def load_data(self, file_path):
        data_list = []
        rate_list = []
        winery_list = []
        genres_list = []
        # y = []
        # users = set()
        # items = set()
        with open(file_path) as f:
            # i = 0
            for line in f:
                # if i == 10:
                #     break
                # else:
                #     i = i + 1
                data = json.loads(line)
                rate = data.pop('rating')
                data['critic_rating'] = str(data['critic_rating'])
                data['audience_rating'] = str(data['audience_rating'])
                data['director'] = str(data['director'])
                data['country'] = str(data['country'])
                # data['genres'] = str(data['genres'])
                data['user'] = str(data['user'])
                data['movie'] = str(data['movie'])

                genres_list.append(set(data['genres'].split('|')))
                data['genres'] = str(genres_list.index(set(data['genres'].split('|'))))
                data_list.append(data)
                rate_list.append(float(rate))
                # winery_list.append(data['winery'])
                # (user, movieid, rating, ts)=line.split('\t')
                # data.append({ "user_id": str(user), "movie_id": str(movieid)})
                # y.append(float(rating))
                #
                # users.add(user)
                # items.add(movieid)
        v = DictVectorizer()
        X = v.fit_transform(data_list)

        # print(X.toarray())

        y = np.array(rate_list)

        return X, y, v, winery_list

    def train(self, file_name):
        X, y, v, winery= self.load_data(file_name)
        # (test_data, y_test, test_users, test_items) = self.load_data("ua.test")
        # v = DictVectorizer()
        X, useless, y, useless = train_test_split(X, y, test_size=0.0, random_state=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        # X_test = v.transform(test_data)

        self.fit(X_train, y_train)

        y_pred = self.output(X_test)
        y_test = y_test.astype('float64')

        print(y_pred[:10])
        print(y_test[:10])
        print('mse:', mean_squared_error(y_test, y_pred))
        print('mae:', mean_absolute_error(y_test, y_pred))

        joblib.dump(self.fm, "model_fm.m")

    def output(self, data):
        y_pre = self.fm.predict(data)
        return y_pre

    def fit(self, x, y):
        self.fm.fit(x, y)


if __name__ == '__main__':
    fm = FM()
    fm.train('/home/next/path/mv/movie_rating')
