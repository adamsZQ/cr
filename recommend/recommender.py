import os

from sklearn.externals import joblib
from surprise import Reader
from surprise import KNNWithMeans
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split


class Recommender:
    def __init__(self, file_prefix, model_path, data_path):
        self.file_prefix = file_prefix
        self.model_path = model_path
        self.data_path = data_path

        self.__load_data()
        self.model = self.__build_model()

    def __load_data(self):
        reader = Reader(sep='\t')
        data = Dataset.load_from_file('{}{}'.format(self.file_prefix, self.data_path), reader=reader)
        trainset, testset = train_test_split(data, test_size=.2, random_state=1)
        self.trainset = trainset
        self.testset = testset

    def __build_model(self):
        model_path = '{}{}'.format(self.file_prefix, self.model_path)
        try:
            model = joblib.load(model_path)
            print('recommender exists, load it')
            return model
        except Exception as e:
            print('recommender does not exist, build new recommender')

            # load data

            # initialize KNN recommender
            algo = KNNWithMeans(k=50, sim_options={'name': 'pearson_baseline', 'user_based': False})
            # train model
            algo.fit(self.trainset)
            # save model
            joblib.dump(algo, model_path)
            # validation
            test_pred = algo.test(self.testset)
            accuracy.rmse(test_pred)

            return algo

    def predict(self, user_id, item_id_list):
        item_list = [self.model.predict(str(user_id), str(item_id))[1] for item_id in item_id_list]
        predictions = [self.model.predict(str(user_id), str(item_id))[3] for item_id in item_id_list]

        return item_list,predictions

    def test(self, testset):
        results = self.model.test(testset)
        return results


if __name__ == '__main__':
    recommender = Recommender('/path/mv/', 'model/knn_model.m', 'ratings_cleaned.dat')

    item_list, predicts = recommender.predict('39320', ['54001'])

    print(item_list, predicts)