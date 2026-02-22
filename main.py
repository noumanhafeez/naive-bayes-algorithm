import numpy as np
from Naive_Bayes_Scratch import NaiveBayesWorkflow
from Naive_Bayes import SklearnNaiveBayesWorkflow
from evaluate_metric import evaluate_metrics


def run_scratch_nb():
    model = NaiveBayesWorkflow("mushrooms.csv")
    # Encode
    model.encode_data()

    # Split
    X_train, X_test, y_train, y_test = model.split_data()

    # Train
    priors = model.calculate_priors(y_train)
    likelihoods = model.calculate_likelihoods(X_train, y_train)

    # Predict
    predictions = model.predict(X_test, priors, likelihoods)

    # Metrics
    metrics = evaluate_metrics(np.array(y_test), predictions)
    print("\nScratch Naive Bayes Metrics:")
    for key, value in metrics.items():
        print(f"{key}:")
        print(value)
        print()

    # Print predictions
    print("Sample Predictions (Scratch Naive Bayes):")
    print(predictions[:20])  # printing first 20 predictions for brevity
    print()


def run_sklearn_nb():
    """Run Sklearn CategoricalNB, display metrics, and predictions."""
    sklearn_model = SklearnNaiveBayesWorkflow("mushrooms.csv")

    # Encode
    sklearn_model.encode_data()

    # Split
    X_train, X_test, y_train, y_test = sklearn_model.split_data()

    # Train
    sklearn_model.train_model(X_train, y_train)

    # Predict
    predictions = sklearn_model.predict(X_test)

    # Metrics
    metrics = evaluate_metrics(y_test, predictions)
    print("\nSklearn CategoricalNB Metrics:")
    for key, value in metrics.items():
        print(f"{key}:")
        print(value)
        print()

    # Print predictions
    print("Sample Predictions (Sklearn CategoricalNB):")
    print(predictions[:20])  # printing first 20 predictions for brevity
    print()


if __name__ == "__main__":
    print("Choose Naive Bayes model to run:")
    print("1. Scratch Naive Bayes")
    print("2. Sklearn CategoricalNB")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        run_scratch_nb()
    elif choice == "2":
        run_sklearn_nb()
    else:
        print("Invalid choice! Please enter 1 or 2.")
