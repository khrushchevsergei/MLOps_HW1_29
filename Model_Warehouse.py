import os
from Basic_Trainer import Basic_Trainer
import dill as pickle


# save the model to the folder

def SaveModel(model_ID, model):
    with open('../models/' + str(model_ID), 'wb') as file:
        pickle.dump(model, file)


# Add new model
def AddModel(model_ID, model_class, Dataset):
    recent_model = Basic_Trainer(model_ID, model_class, Dataset)
    SaveModel(model_ID, recent_model)


# load the model from the directory

def LoadModel(model_ID):
    with open('../models/' + str(model_ID), 'rb') as file:
        model = pickle.load(file)
    return model


def FitModel(model_ID, model_params=None):
    model = LoadModel(model_ID)
    model.fit(model_params=model_params)
    SaveModel(model_ID, model)


# Generate prediction on Dataset
def MakePrediction(model_ID):
    model = LoadModel(model_ID)
    prediction = model.predict()
    return prediction


# Delete the model
def DeleteModel(model_ID):
    os.remove('../models/' + str(model_ID))
