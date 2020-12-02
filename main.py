
# A very simple Flask Hello World app for you to get started with...

from flask import Flask, request, jsonify, render_template

import numpy as np
import pandas as pd


from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

data = pd.read_csv("https://raw.githubusercontent.com/sjoerdteunisse/COVID-Awareness/main/data.csv")

X = data.drop('PossibileInfectionChance', axis=1)
y = data['PossibileInfectionChance']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=9)
lr = LinearRegression()
lr.fit(X_train, y_train)

@app.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
  return response


@app.route("/api/v1/dataset")
def jsonTest():
    return jsonify({"test": data.to_json()})

@app.route('/api/v1/predict', methods=['GET','POST'])
def parse_request():
    data = request.json

    predictions = lr.predict([[ int(data["pI"]) , int(data["sH"]) , int(data["hI"]), int(data["wH"]), int(data["m"]), int(data["sD"]) ]])
    response = jsonify({"result": predictions[:10][0]})

    return response

