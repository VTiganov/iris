import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import accuracy_score, classification_error, confusion_matrix

iris = load_iris()

iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target

iris_df['species'] = iris_df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# print(iris_df.describe())
# print("Dataset shape:", iris_df.shape)
# print(iris_df.isnull().sum())
# CHECKING DATAFRAME

X = iris_df.drop('species', axis=1)
y = iris_df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print("Training set shape:", X_train.shape)
# print("Testing set shape:", X_test.shape)
# 20% TEST 80% TRAIN
