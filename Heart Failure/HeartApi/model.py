import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import MinMaxScaler
import pickle
import joblib 


# load dataset
data = pd.read_csv('heart.csv') 

def prepare_data(df):

    # create dummies
    cat_cols = ['Sex', 'ChestPainType','RestingECG','ExerciseAngina', 'ST_Slope']

    df = pd.get_dummies(df, columns=cat_cols, prefix_sep = ':')

    return df

df = prepare_data(data)

def split_normalize_data(data):

    # splitting the data
    X = data.drop('HeartDisease',axis=1)
    y = data['HeartDisease']

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=102)

    # fit scaler on training data
    normalizer = MinMaxScaler().fit(X_train)

    # transform training data
    X_train_norm = normalizer.transform(X_train)

    # transform testing data
    X_test_norm = normalizer.transform(X_test)

    return X_train_norm, X_test_norm, y_train, y_test

X_train_norm = split_normalize_data(df)[0]
X_test_norm = split_normalize_data(df)[1]
y_train = split_normalize_data(df)[2]
y_test = split_normalize_data(df)[3]

# modeling
classifier = LogisticRegression()
classifier.fit(X_train_norm, y_train)

# save the model
with open('model.pkl', 'wb') as files:
    pickle.dump(classifier, files)
    print("Model successfully saved")

# saving the columns
with open('model_columns.pkl', 'wb') as files:
    X = df.drop('HeartDisease',axis=1)
    model_columns = list(X.columns)
    pickle.dump(model_columns, files)