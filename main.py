"""
CSCE-478-700
ML taxi price analysis project
Authors: Sasha Gerasimov, Johnathan Franklin, Grant Henderson

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, ConfusionMatrixDisplay,
)

#1. Loading the data
print("STEP 1: Loading the data")
df = pd.read_csv("cab_rides.csv").dropna() # drop rows with missing values
print("After dropping missing values, the dataset has {} rows and {} columns.".format(df.shape[0], df.shape[1]))
print("The columns in the dataset are: \n", df.dtypes)


#2. Data cleaning and preprocessing
print("STEP 2: Data cleaning and preprocessing")
print("\"product_id\" and \"id\" columns is not needed for analysis, dropping it.")
df.drop("product_id", axis=1, inplace=True)
df.drop("id", axis=1, inplace=True)
print("The column \"cab_type\" only contains two values: Lyft and Uber. We will convert it to a binary variable for analysis. Lyft will be represented as 0 and Uber as 1.")
df['cab_type_binary'] = df["cab_type"].map({'Lyft': 0, 'Uber': 1})

df["datetime"] = pd.to_datetime(df["time_stamp"], unit="ms")
df["hour"] = df["datetime"].dt.hour
df["dayofweek"] = df["datetime"].dt.dayofweek
print("Created 'hour' and 'dayofweek' from time_stamp.")
print("Now, the columns in the dataset are: \n", df.dtypes)

print("The first 5 rows of the cleaned dataset are: \n", df.head())

#3. Feature selection and splitting the data into training and testing sets
print("STEP 3: Feature selection and splitting the data into training and testing sets")

X = df.drop(columns=["price", "cab_type", "time_stamp"])
y = df[["price"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#4. Model training and evaluation
print("STEP 4: Model training and evaluation")
print("We will train abd evaluate Linear Regression and Random Forests models on the dataset.")
# Define the models to be evaluated


print("ran successfully") # keep at the end of the file to ensure that the code runs without errors, delete before submission
