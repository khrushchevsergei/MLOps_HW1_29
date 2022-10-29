# MLOps_HW1_29
 
import requests
import json
import pandas as pd

# Add data 

Data = pd.read_csv('./Dataset/train_flats_prices.csv')
JData = Data.to_json()

#Add model 

headers = {"accept": "application/json", "content-type": "application/json"}
params = {"model_ID": 1, "model_class": "LogisticRegression", "Dataset": JData}
r = requests.post('http://127.0.0.1:5000/model/add', headers=headers, data=json.dumps(params))
print(r, r.text)

headers = {"accept": "application/json", "content-type": "application/json"}
params = {"model_ID": 2, "model_class": "GradientBoostingClassifier", "Dataset": JData}
r = requests.post('http://127.0.0.1:5000/model/add', headers=headers, data=json.dumps(params))
print(r, r.text)

# Train model 

model_params = {'random_state': 10}
params = {"model_ID": 1, "model_params" : model_params}
r = requests.post('http://127.0.0.1:5000/model/fit', headers=headers, data=json.dumps(params))
print(r, r.text)

# Get predictoin 

params = {"model_ID" : 1}
r = requests.get('http://127.0.0.1:5000/model/predict', headers=headers, data = json.dumps(params))
print(r, r.text)


# Saved models 

r = requests.get('http://127.0.0.1:5000/available_models')
print(r, r.text)

# Delete model 

params = {"model_ID" : 1}
r = requests.delete('http://127.0.0.1:5000/model/delete', headers=headers, data = json.dumps(params))
print(r, r.text)

r = requests.get('http://127.0.0.1:5000/available_models', headers=headers)
print(r, r.text)