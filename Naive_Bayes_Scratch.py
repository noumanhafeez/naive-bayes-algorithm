import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



class NaiveBayesWorkflow:

    def __init__(self, filepath):
        # Load dataset
        self.data = pd.read_csv(filepath)
        self.encoders = {}

    ## Encoding
    def encode_data(self):
        for col in self.data.columns:
            encoder = LabelEncoder()
            self.data[col] = encoder.fit_transform(self.data[col])
            self.encoders[col] = encoder
        return self.data

    ## Train Test Split
    def split_data(self):
        X = self.data.drop('class', axis=1)
        y = self.data['class']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    ## Priors Probabilities
    def calculate_priors(self, y_train):
        classes, counts = np.unique(y_train, return_counts=True)
        total = len(y_train)

        priors = {}
        for c, count in zip(classes, counts):
            priors[c] = np.log(count / total)  # Log Prior

        return priors

    ## Likelihoods (Laplace Smoothing)
    def calculate_likelihoods(self, X_train, y_train):

        X = np.array(X_train)
        y = np.array(y_train)

        classes = np.unique(y)
        n_features = X.shape[1]

        likelihoods = {}
        feature_unique_values = {}

        # Count unique values per feature
        for i in range(n_features):
            feature_unique_values[i] = len(np.unique(X[:, i]))

        for c in classes:
            likelihoods[c] = {}
            X_c = X[y == c]
            N_c = len(X_c)

            for i in range(n_features):
                likelihoods[c][i] = {}

                K = feature_unique_values[i]
                values, counts = np.unique(X_c[:, i], return_counts=True)

                for v in np.unique(X[:, i]):
                    count = counts[values.tolist().index(v)] if v in values else 0
                    prob = (count + 1) / (N_c + K)  # Laplace smoothing
                    likelihoods[c][i][v] = np.log(prob)

        return likelihoods

    ## Prediction
    def predict(self, X_test, priors, likelihoods):

        X_test = np.array(X_test)
        predictions = []

        for sample in X_test:
            class_scores = {}

            for c in priors:
                log_prob = priors[c]

                for i, value in enumerate(sample):
                    log_prob += likelihoods[c][i][value]

                class_scores[c] = log_prob

            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)

        return np.array(predictions)