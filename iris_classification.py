import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()

iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target

iris_df['species'] = iris_df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# print(iris_df.describe())
# print("Dataset shape:", iris_df.shape)
# print(iris_df.isnull().sum())
# CHECKING DATAFRAME