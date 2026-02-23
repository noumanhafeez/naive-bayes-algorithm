# 🍄 Naive Bayes Classifier – From Scratch & Sklearn
#
# This project implements the Naive Bayes classification algorithm in two different ways.
#
# 1. From Scratch (Custom Implementation using NumPy)
# 2. Using Scikit-Learn (CategoricalNB)
#
# The model is trained on the Mushroom Dataset to classify mushrooms as:
#
# Edible
# Poisonous
#
# The project also supports prediction from a custom CSV file.
#
# ------------------------------------------------------------
# Project Structure
# ------------------------------------------------------------
#
# naive-bayes-algorithm/
# ├── mushrooms.csv
# ├── mushroom_input.csv
# ├── Naive_Bayes_Scratch.py
# ├── Naive_Bayes.py
# ├── evaluate_metric.py
# ├── main.py
# └── README.md
#
# ------------------------------------------------------------
# Implementation Details
# ------------------------------------------------------------
#
# 1. Naive Bayes From Scratch
#
# Implemented inside:
# Naive_Bayes_Scratch.py
#
# Step 1: Data Encoding
# All categorical features are converted into numeric values using LabelEncoder.
# Encoders are stored in self.encoders so they can be reused during prediction.
#
# Step 2: Train-Test Split
# train_test_split(test_size=0.2, random_state=42)
# 80 percent data is used for training.
# 20 percent data is used for testing.
#
# Step 3: Prior Probabilities
# P(c) = count(c) / total_samples
# Priors are stored in log form using np.log() to avoid numerical underflow.
#
# Step 4: Likelihood Probabilities with Laplace Smoothing
# P(x_i | c) = (count + 1) / (N_c + K)
# N_c is the number of samples in class c.
# K is the number of unique feature values.
# Laplace smoothing prevents zero probability issues.
# Likelihoods are stored as log probabilities.
#
# Step 5: Prediction Rule
# For each sample:
# log P(c) + sum of log P(x_i | c)
# The class with the highest score is selected.
#
# ------------------------------------------------------------
# 2. Sklearn Implementation
# ------------------------------------------------------------
#
# Implemented inside:
# Naive_Bayes.py
#
# Uses:
# from sklearn.naive_bayes import CategoricalNB
#
# Workflow:
# Encode dataset
# Split dataset
# Train model
# Predict
# Evaluate performance
#
# ------------------------------------------------------------
# Evaluation Metrics
# ------------------------------------------------------------
#
# Implemented inside:
# evaluate_metric.py
#
# Metrics included:
# Accuracy
# Precision
# Recall
# F1-Score
# Confusion Matrix
#
# ------------------------------------------------------------
# How To Run The Project
# ------------------------------------------------------------
#
# Step 1: Install Required Libraries
# pip install pandas numpy scikit-learn
#
# Step 2: Run the project
# python main.py
#
# You will see:
# 1. Run Naive Bayes on train/test split
# 2. Predict from custom CSV
#
# Choose your desired option.
#
# ------------------------------------------------------------
# Predict From Custom CSV
# ------------------------------------------------------------
#
# Example file:
# mushroom_input.csv
#
# Important Rules:
# The file must contain only feature columns.
# The file must NOT contain the target column (class).
# Column names must match the training dataset.
# Categories must already exist in training data.
# Encoding from training must be reused.
#
# Example Output:
# Sample 1: Edible
# Sample 2: Poisonous
#
# ------------------------------------------------------------
# Important Notes
# ------------------------------------------------------------
#
# Column order must match the training dataset.
# Log probabilities are used for numerical stability.
# Laplace smoothing prevents zero probability issues.
# Custom prediction reuses saved encoders.
#
# ------------------------------------------------------------
# Key Learning Outcomes
# ------------------------------------------------------------
#
# Implemented Naive Bayes from scratch.
# Applied Laplace smoothing.
# Used log probabilities for stability.
# Built Scikit-Learn version.
# Created custom CSV prediction system.
# Designed modular ML workflow.
#
# ------------------------------------------------------------
# Future Improvements
# ------------------------------------------------------------
#
# Add probability confidence scores.
# Handle unseen categories safely.
# Save predictions to CSV.
# Convert to CLI tool.
# Build Streamlit Web App.
#
# ------------------------------------------------------------
# Author
# ------------------------------------------------------------
#
# Nouman Hafeez
# Machine Learning Enthusiast