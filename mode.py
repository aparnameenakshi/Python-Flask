import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# Load the dataset from an Excel file
data = pd.read_excel('C:/Users/aksha/Desktop/iris/iris .xls')


# Assuming the last column is the target variable and other columns are features
X = data.iloc[:, :-1]  # Selecting all columns except the last one as features
y = data.iloc[:, -1]   # Selecting the last column as the target variable

# Train a machine learning model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save the trained model as a .pkl file
with open('trained_model.pkl', 'wb') as file:
    pickle.dump(model, file)
