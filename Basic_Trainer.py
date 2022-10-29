import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


def dataset_prepare(Dataset):
    # Dataset : pd.DataFrame
    # Dataset from './Dataset/train_flats_prices.csv'

    Dataset = Dataset.drop(columns=Dataset.dtypes[Dataset.dtypes == "object"].index)
    Dataset = Dataset.drop(columns=["Id"])

    X = Dataset.drop(columns=["SalePrice"])
    y = Dataset["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

    numeric_data = X_train.select_dtypes([np.number])
    numeric_data_mean = numeric_data.mean()
    numeric_features = numeric_data.columns

    X_train = X_train.fillna(numeric_data_mean)
    X_test = X_test.fillna(numeric_data_mean)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[numeric_features])
    X_test_scaled = scaler.transform(X_test[numeric_features])
    return X_train, X_test, y_train, y_test


class Basic_Trainer:
    # model_ID : int
    # model_Classs: srt
    # data :pd.DataFrame from  './Dataset/train_flats_prices.csv'

    def __init__(self, model_ID, model_class, Dataset):
        models = {'GradientBoostingClassifier': GradientBoostingClassifier(),
                  'LogisticRegression': LogisticRegression()}
        self.model_ID = model_ID
        self.model_class = model_class
        self.model = models[self.model_class]
        self.X_train, self.X_test, self.y_train, self.y_test = dataset_prepare(Dataset)

    # fit the model
    def fit(self, model_params):
        self.model.set_params(**model_params)
        self.model.fit(self.X_train, self.y_train)

    # get predict on the Dataset
    def predict(self):
        prediction = self.model.predict(self.X_test)
        return prediction

    # get the model parameters
    def get_params(self):
        return self.model.get_params()