import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException
import pickle

import lightgbm as lgb
from lightgbm import LGBMClassifier
import joblib

app = FastAPI() #check query parameters

final_dataframe = pd.read_csv("D:/Downloads/final_dataframe.csv")
predict_df = pd.read_csv("D:/Downloads/predict.csv")
'''best_lgb = joblib.load('D:/Downloads/best_lightgbm_model.pkl')'''

@app.get("/")
async def root():
    return "Vérification d'enregistrement"

'''@app.get("/liste_client")
async def liste_client():
    return predict_df['sk-id-curr'].values'''

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