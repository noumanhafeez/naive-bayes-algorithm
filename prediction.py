from Naive_Bayes import SklearnNaiveBayesWorkflow
import pandas as pd

sklearn_model = SklearnNaiveBayesWorkflow('mushrooms.csv')
sklearn_model.encode_data()

X_train, X_test, y_train, y_test = sklearn_model.split_data()
sklearn_model.train_model(X_train, y_train)

user_data = pd.read_csv("mushroom_input.csv")
user_data.drop(columns=['type'], axis=1, inplace=True)


for column in user_data.columns:
    if column in sklearn_model.encoders:
        user_data[column] = sklearn_model.encoders[column].transform(user_data[column])

predictions = sklearn_model.predict(user_data)

label_map = {
    0: "Edible 🍄",
    1: "Poisonous ☠️"
}

print("\nPredictions for Custom CSV:\n")

for i, pred in enumerate(predictions):
    print(f"Sample {i+1}: {label_map[pred]}")