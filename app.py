import os
import joblib
import json
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from peewee import (
    SqliteDatabase, PostgresqlDatabase, Model, IntegerField,
    FloatField, TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict


# unpickle the previously-trained sklearn model
with open('columns.json') as fh:
    columns = json.load(fh)

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)

pipeline = joblib.load('pipeline.pickle')


# set up database
DB = SqliteDatabase('predictions.db')
class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    proba = FloatField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB

DB.create_tables([Prediction], safe=True)


# create Flask webserver
app = Flask(__name__)




@app.route('/predict', methods=['POST'])
def predict():
    """
    Produce prediction for request.
    
    Inputs:
        request: dictionary with format described below
        
        ```
        {
            "observation_id": <id-as-a-string>,
            "data": {
                "age": <value>,
                "sex": <value>,
                "cp": <value>,
                "trestbps": <value>,
                "fbs": <value>,
                "restecg": <value>,
                "oldpeak": <value>,
                "ca": <value>,
                "thal": <value>
            }
        }
        ```
     
    Returns: A dictionary with predictions or an error, the two potential values:
                ```
                {
                    "observation_id": <id-of-request>,
                    "prediction": <True|False>,
                    "probability": <probability generated by model>
                }
                ```
                or 
                ```
                {
                    "observation_id": <id-of-request>,
                    "error": "some error message"
                }
                ```
                if success is False, return an error string
    """

    req = request.get_json()
    
    # check if request has observation_id
    if 'observation_id' not in req:
        response = {'observation_id': None,
                    'error': 'Must supply observation_id'}
        return jsonify(response)
    
    # check if request has data
    if 'data' not in req:
        response = {'observation_id': req['observation_id'],
                    'error': 'Must supply data'}
        return jsonify(response)
    
    # check if request data has all necessary columns
    necessary_columns = {'age', 'ca', 'cp', 'fbs', 'oldpeak', 'restecg', 'sex', 'thal', 'trestbps'}
    actual_columns = set(req['data'].keys())
    if not necessary_columns.issubset(actual_columns):
        missing_columns = necessary_columns - actual_columns
        response = {'observation_id': req['observation_id'],
                    'error': 'Missing columns: {}'.format(missing_columns)}
        return jsonify(response)
    
    # check if request data has extra columns
    if not actual_columns.issubset(necessary_columns):
        extra_columns = actual_columns - necessary_columns
        response = {'observation_id': req['observation_id'],
                    'error': 'Unrecognized columns provided: {}'.format(extra_columns)}
        return jsonify(response)
    
    # check sex data
    valid_sex = [0, 1]
    sex = req['data']['sex']
    if sex not in valid_sex:
        response = {'observation_id': req['observation_id'],
                    'error': 'Invalid value provided for sex: {}. Allowed values are: {}'.format(sex, valid_sex)}
        return jsonify(response)
    
    # check ca data
    valid_ca = [0, 1, 2, 3]
    ca = req['data']['ca']
    if ca not in valid_ca:
        response = {'observation_id': req['observation_id'],
                    'error': 'Invalid value provided for ca: {}. Allowed values are: {}'.format(ca, valid_ca)}
        return jsonify(response)
    
    # check age data
    age = req['data']['age']
    if not 0 <= age < 150:
        response = {'observation_id': req['observation_id'],
                    'error': 'Invalid value provided for age: {}. Needs to be in [0, 150).'.format(age)}
        return jsonify(response)
    
    # check trestbps data
    trestbps = req['data']['trestbps']
    if not 50 <= trestbps < 300:
        response = {'observation_id': req['observation_id'],
                    'error': 'Invalid value provided for trestbps: {}. Needs to be in [50, 300)'.format(trestbps)}
        return jsonify(response)
    
    # check oldpeak data
    oldpeak = req['data']['oldpeak']
    if not oldpeak < 10:
        response = {'observation_id': req['observation_id'],
                    'error': 'Invalid value provided for oldpeak: {}. Needs to smaller than 10'.format(oldpeak)}
        return jsonify(response)

    # initialise input data
    X = pd.DataFrame([req['data']], index=[req['observation_id']], columns=columns).astype(dtypes)
    
    # create output
    response = {'observation_id': req['observation_id'],
                'prediction': bool(pipeline.predict(X)[0]),
                'probability': pipeline.predict_proba(X)[0, 1]}

    # store
    p = Prediction(
        observation_id=response['observation_id'],
        proba=response['probability'],
        observation=req['data'],
    )
    try:
        p.save()
    except IntegrityError:
        error_msg = "ERROR: Observation ID: '{}' already exists".format(_id)
        response["error"] = error_msg
        print(error_msg)
        DB.rollback()

    return jsonify(response)

# def predict():
#     # deserialise data
#     raw = request.get_json()
#     _id = raw['id']
#     data = raw['observation']

#     # create dataframe
#     try:
#         X = pd.DataFrame([data], columns=columns).astype(dtypes)
#     except ValueError:
#         error_msg = 'Observation is invalid!'
#         response = {'error': error_msg}
#         print(error_msg)
#         return jsonify(response)

#     # predict
#     proba = pipeline.predict_proba(X)[0, 1]
#     response = {'proba': proba}
#     p = Prediction(
#         observation_id=_id,
#         proba=proba,
#         observation=request.data
#     )
#     try:
#         p.save()
#     except IntegrityError:
#         error_msg = 'Observation ID: "{}" already exists'.format(_id)
#         response['error'] = error_msg
#         print(error_msg)
#         DB.rollback()
#     return jsonify(response)


@app.route('/update', methods=['POST'])
def update():
    raw = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == raw['observation_id'])
        p.true_class = raw['true_class']
        p.save()
        return jsonify(model_to_dict(p))
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(raw['observation_id'])
        return jsonify({'error': error_msg})


@app.route('/list-db-contents')
def list_db_contents():
    return jsonify([
        model_to_dict(obs) for obs in Prediction.select()
    ])

# run
if __name__ == "__main__":
    app.run(debug=True)