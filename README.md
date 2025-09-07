💳 Credit Card Fraud Detection using Machine Learning

A Machine Learning project to detect fraudulent credit card transactions using Logistic Regression. This model is trained on the popular Credit Card Fraud Detection Dataset and achieves 92%+ accuracy on test data.

📌 Problem Statement

Credit card fraud is a serious issue in the financial sector, leading to billions in losses every year. Detecting fraudulent transactions in real time is crucial to protect both customers and businesses.

⚡ Solution

This project builds a binary classification model that classifies transactions as:

0 → Normal Transaction

1 → Fraudulent Transaction

Key steps:

Data preprocessing & undersampling to balance the dataset.

Train-test split with stratification.

Logistic Regression model training.

Evaluation with Accuracy, Precision, Recall, and F1-score.

📊 Results
Training Data (787 samples)

Accuracy: 95.43%

Precision: [0.93, 0.99]

Recall: [0.99, 0.92]

F1-score: ~0.95

Testing Data (197 samples)

Accuracy: 92.89%

Precision: [0.90, 0.96]

Recall: [0.96, 0.89]

F1-score: ~0.93

✅ The model achieves a strong balance between precision and recall, making it effective for fraud detection.

🛠️ Tech Stack

Python

NumPy, Pandas

Scikit-learn

📥 Dataset

The dataset used in this project is available on Kaggle:
[🔗 Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)


🎥 Demo Video

[Add demo video link here]

📜 License

This project is licensed under the MIT License — free to use and modify with attribution.
