from flask import Flask, request, jsonify, Response
import pickle
import pandas as pd
import numpy as np
import traceback
import joblib
from sklearn.preprocessing import MinMaxScaler
from model import prepare_data

# load model
with open('model.pkl', 'rb') as files:
    model = pickle.load(files)

# load model columns
with open('model_columns.pkl', 'rb') as files:
    model_columns = pickle.load(files)


app = Flask(__name__)

@app.route('/', methods=['POST'])

def predict():
    json_ = request.json

    user_df = pd.DataFrame.from_dict([json_])
    query_df = prepare_data(user_df)

    for col in model_columns:
        if col not in query_df.columns:
            query_df[col] = 0

    scaler = MinMaxScaler().fit(query_df)
    query_df = scaler.transform(query_df)

    prediction = model.predict(query_df)
    prediction = np.where((prediction == 1),"Heart Disease","Normal")

    return jsonify({"Patient Status": str(prediction)})

if __name__ == '__main__':
    app.run(port=8000, debug=True,use_reloader=False)