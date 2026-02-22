# naive_bayes.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB



class SklearnNaiveBayesWorkflow:

    def __init__(self, filepath):
        # Load dataset
        self.data = pd.read_csv(filepath)
        self.encoders = {}

    ## Encode categorical features
    def encode_data(self):
        for col in self.data.columns:
            encoder = LabelEncoder()
            self.data[col] = encoder.fit_transform(self.data[col])
            self.encoders[col] = encoder
        return self.data

    ## Split into train/test
    def split_data(self, test_size=0.2, random_state=42):
        X = self.data.drop('class', axis=1)
        y = self.data['class']
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    ## Train sklearn CategoricalNB
    def train_model(self, X_train, y_train):
        self.model = CategoricalNB()
        self.model.fit(X_train, y_train)
        return self.model

    ## Predict
    def predict(self, X_test):
        return self.model.predict(X_test)
