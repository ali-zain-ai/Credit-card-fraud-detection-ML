import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score

data_set = pd.read_csv(r"D:\AI_Python\Credit Card check model\creditcard.csv")

# print(data_set.head(1rain

# print(data_set.info())

# print(data_set['Class'].value_counts())

# In this dataset:
# 0---> Normal transection
# 1---> Furad Transection

# Now we store the 0 and 1 in seperate variables
Legit = data_set[data_set.Class == 0]
Furad = data_set[data_set.Class == 1]

# print(Legit.shape) #(284315, 31)
# print(Furad.shape) #(492, 31)

#Now compair the value of both transections
# print(data_set.groupby('Class').mean())

#Under sampling
#Build a sample dataset containing the similar distribution of normal and fraud transections

legit_sample = Legit.sample(n = 492)

#Concatinating two dataframes

new_dataset = pd.concat([legit_sample, Furad], axis=0)
# print(new_dataset.head(10))
# print(new_dataset['Class'].value_counts())
# print(new_dataset.groupby('Class').mean())


#spliting the features and targets(Lables)

X = new_dataset.drop(columns='Class', axis = 1)
# print(X)
Y = new_dataset['Class']
# print(Y)

# Spliting the training and testing data
X_train, X_test, Y_train, Y_test = (train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=1))

#training the model 
model = LogisticRegression(max_iter=9000)
model.fit(X_train, Y_train)

y_pred = model.predict(X_train)

accuracy = accuracy_score(Y_train, y_pred)
precision = precision_score(Y_train, y_pred, average=None)
recall = recall_score(Y_train, y_pred, average=None)

print(f"Model Accuracy: {accuracy*100:.2f}%")
print(f"Precision_Score = {precision}")
print(f"Recall_Score = {recall}")

print("Classification_Report\n",classification_report(Y_train, y_pred))