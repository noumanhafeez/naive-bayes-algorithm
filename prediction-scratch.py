import pandas as pd
from Naive_Bayes_Scratch import NaiveBayesWorkflow

model = NaiveBayesWorkflow("mushrooms.csv")

model.encode_data()

X_train, X_test, y_train, y_test = model.split_data()

priors = model.calculate_priors(y_train)
likelihoods = model.calculate_likelihoods(X_train, y_train)

user_data = pd.read_csv("mushroom_input.csv")

# If your custom file has 'type' column remove it
if 'type' in user_data.columns:
    user_data.drop(columns=['type'], inplace=True)


for column in user_data.columns:
    if column in model.encoders:
        user_data[column] = model.encoders[column].transform(user_data[column])

predictions = model.predict(user_data, priors, likelihoods)

print("\nScratch Naive Bayes Predictions:\n")

for i, pred in enumerate(predictions):
    if pred == 0:
        print(f"Sample {i+1}: Edible 🍄")
    else:
        print(f"Sample {i+1}: Poisonous ☠️")