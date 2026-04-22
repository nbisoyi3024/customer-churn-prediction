import os
import joblib


#save the model as pickle file
def save_model(model, path):
    joblib.dump(model,path)

#Load the model  from a pickle file
def load_model(path):
    return joblib.load(path)