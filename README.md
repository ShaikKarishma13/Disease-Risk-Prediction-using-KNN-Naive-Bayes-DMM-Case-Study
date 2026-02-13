🫀 Disease Risk Prediction using KNN & Naive Bayes

A Machine Learning project developed as part of the Data Mining & Machine Learning (DMM) course to predict disease risk using real-world heart disease data.
This project implements and compares K-Nearest Neighbors (KNN) and Gaussian Naive Bayes models, evaluates their performance using multiple metrics, and provides real-time risk prediction for new patient inputs.

📌 Project Overview
The goal of this project is to:
Preprocess real medical dataset
Perform feature engineering
Generate risk labels using clustering
Train and compare classification models
Evaluate model performance using multiple metrics
Build an interactive prediction system

📊 Dataset
Source: Public heart disease dataset
Features used:
Age
Resting Blood Pressure (BP)
Maximum Heart Rate
Fasting Blood Sugar (used as Fever indicator)
Exercise-Induced Angina (used as Cough indicator)

⚙️ Methodology

1️⃣ Data Preprocessing
Loaded dataset using Pandas
Selected relevant features
Converted categorical values to numeric
Train-test split (70% – 30%)
2️⃣ Risk Label Generation
Applied K-Means Clustering (7 clusters)
Converted cluster labels into binary risk (Low / High)
3️⃣ Feature Scaling
Used StandardScaler for KNN (distance-based model)
4️⃣ Model Training
K-Nearest Neighbors (KNN)
Gaussian Naive Bayes
5️⃣ Hyperparameter Tuning
Tested K values from 1–20
Selected best K based on highest accuracy
Best K = 7
6️⃣ Model Evaluation Metrics
Accuracy
Precision
Recall
F1-Score
ROC Curve
AUC (Area Under Curve)

📈 Results
The performance of both models is as follows:
KNN Model
Accuracy: 92.6%
F1 Score: 0.91
AUC: 0.98
Naive Bayes Model
Accuracy: 86.6%
F1 Score: 0.83
AUC: 0.96

🔍 Observations
KNN outperformed Naive Bayes in terms of Accuracy and AUC.
Feature scaling significantly improved the performance of the KNN model.
ROC curve comparison provided a more reliable evaluation beyond just accuracy.
AUC scores indicate both models perform well, but KNN shows stronger class separation capability.

💻 Real-Time Prediction
The project includes a user input prediction system where new patient data can be entered manually:
Enter Age
Enter Blood Pressure
Enter Heart Rate
Fever? (0 = No, 1 = Yes)
Cough? (0 = No, 1 = Yes)
Based on the input values, the system predicts:
High Risk
Low Risk
Predictions are generated using both KNN and Naive Bayes models.

🛠 Technologies & Tools
Python
Pandas
NumPy
Scikit-Learn
Matplotlib
Google Colab

👩‍💻 Team Members
Pedaprolu S S L Katyayani
Vibudhi sahithi

 🚀 Key Learnings
Importance of feature scaling in machine learning models
Model comparison beyond accuracy
Hyperparameter tuning for optimal performance
Understanding and interpreting ROC-AUC
Implementing an end-to-end machine learning workflow
