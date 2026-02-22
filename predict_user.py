# predict_user.py

import pandas as pd
import numpy as np
from Naive_Bayes_Scratch import NaiveBayesWorkflow
from Naive_Bayes import SklearnNaiveBayesWorkflow

def create_label_encoders(train_df):

    encoders = {}
    for col in train_df.columns:
        encoders[col] = {val: idx for idx, val in enumerate(train_df[col].unique())}
    return encoders

def encode_df(df, encoders):

    encoded_df = df.copy()
    for col in df.columns:
        if col in encoders:
            encoded_df[col] = df[col].map(lambda x: encoders[col].get(x, -1))  # unknowns as -1
    return encoded_df.values

def safe_predict(model, X_user_encoded, priors, likelihoods, epsilon=1e-6):

    predictions = []
    classes = list(priors.keys())

    for row in X_user_encoded:
        class_probs = {}
        for c in classes:
            # Use small value if prior is zero
            prior = priors.get(c, epsilon)
            log_prob = np.log(prior if prior > 0 else epsilon)

            for i, value in enumerate(row):
                # Use small value if likelihood missing or zero
                if i in likelihoods[c] and value in likelihoods[c][i]:
                    prob = likelihoods[c][i].get(value, epsilon)
                else:
                    prob = epsilon  # unseen value

                # Avoid log(0)
                prob = prob if prob > 0 else epsilon
                log_prob += np.log(prob)

            class_probs[c] = log_prob

        # choose class with max probability
        pred_class = max(class_probs, key=class_probs.get)
        predictions.append(pred_class)

    return predictions

# ---------------- Prediction Functions ---------------- #

def predict_scratch(user_csv):
    """Predict using Scratch Naive Bayes on user CSV"""
    # Load training data
    train_df = pd.read_csv("mushrooms.csv")
    X_train = train_df.drop("class", axis=1)
    y_train = train_df["class"]

    # Create encoders and encode training data
    encoders = create_label_encoders(X_train)
    X_train_encoded = encode_df(X_train, encoders)

    # Train Scratch Naive Bayes
    model = NaiveBayesWorkflow("mushrooms.csv")
    model.encode_data()
    priors = model.calculate_priors(y_train)
    likelihoods = model.calculate_likelihoods(X_train_encoded, y_train.values)

    # Load and encode user CSV
    user_data = pd.read_csv(user_csv)
    X_user_encoded = encode_df(user_data, encoders)

    # Predict safely
    predictions = safe_predict(model, X_user_encoded, priors, likelihoods)
    user_data['Predicted_Class'] = predictions

    print("\nScratch Naive Bayes Predictions:")
    print(user_data)
    user_data.to_csv("user_predictions_scratch.csv", index=False)
    print("\nPredictions saved to user_predictions_scratch.csv\n")


def predict_sklearn(user_csv):

    # Load training data
    train_df = pd.read_csv("mushrooms.csv")
    X_train = train_df.drop("class", axis=1)
    y_train = train_df["class"]

    # Create encoders and encode training data
    encoders = create_label_encoders(X_train)
    X_train_encoded = encode_df(X_train, encoders)

    # Train Sklearn model
    model = SklearnNaiveBayesWorkflow("mushrooms.csv")
    model.encode_data()
    model.train_model(X_train_encoded, y_train.values)

    # Load and encode user CSV
    user_data = pd.read_csv(user_csv)
    X_user_encoded = encode_df(user_data, encoders)

    # Predict
    predictions = model.predict(X_user_encoded)
    user_data['Predicted_Class'] = predictions

    print("\nSklearn CategoricalNB Predictions:")
    print(user_data)
    user_data.to_csv("user_predictions_sklearn.csv", index=False)
    print("\nPredictions saved to user_predictions_sklearn.csv\n")



if __name__ == "__main__":
    print("Choose model for user CSV prediction:")
    print("1. Scratch Naive Bayes")
    print("2. Sklearn CategoricalNB")
    choice = input("Enter 1 or 2: ").strip()
    user_csv = input("Enter your CSV file path (without target column): ").strip()

    if choice == "1":
        predict_scratch(user_csv)
    elif choice == "2":
        predict_sklearn(user_csv)
    else:
        print("Invalid choice! Please enter 1 or 2.")