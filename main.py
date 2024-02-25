import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException
import pickle

import lightgbm as lgb
from lightgbm import LGBMClassifier
import joblib
import json

import gc
from contextlib import contextmanager
import time
from lightgbm import LGBMClassifier
import warnings
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score


app = FastAPI() #check query parameters

final_dataframe = pd.read_csv("D:/Downloads/final_dataframe.csv")
predict_df = pd.read_csv("D:/Downloads/predict.csv")
best_lgb = joblib.load('D:/Downloads/best_lightgbm_model.pkl')

@app.get("/")
async def root():
    return "Vérification d'enregistrement"

@app.get("/get_unique_client_ids")
async def get_unique_client_ids():
    unique_client_ids = predict_df['sk-id-curr'].unique().tolist()
    return unique_client_ids

@app.get("/prediction_client")
async def prediction_client(client_id: int):
    # Vérifier si le client_id est présent dans predict_df['sk-id-curr']
    if client_id not in predict_df['sk-id-curr'].values:
        raise HTTPException(status_code=404, detail="Client ID not found")
    
    # Obtenir la ligne correspondante du DataFrame predict_df
    client_row = predict_df[predict_df['sk-id-curr'] == client_id]
    
    # Convertir la ligne en dictionnaire pour le retour
    client_data = client_row.to_dict(orient='records')[0]
    
    return client_data

@app.get("/prediction_client_live")
async def prediction_client_live(client_id: int):
    # Vérifier si le client_id est présent dans predict_df['sk-id-curr']
    if client_id not in predict_df['sk-id-curr'].values:
        raise HTTPException(status_code=404, detail="Client ID not found")

    # Obtenir les données du client correspondant du DataFrame predict_df
    client_data = final_dataframe[final_dataframe['sk-id-curr'] == client_id].drop(columns=['target', 'sk-id-curr', 'index'])

    # Effectuer la prédiction en direct
    prediction = best_lgb.predict(client_data).tolist()
    prediction_dict = {'prediction': prediction}

    # Vous pouvez retourner le résultat de la prédiction comme vous le souhaitez, par exemple :
    return prediction_dict