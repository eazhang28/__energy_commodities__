import requests
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing._encoders import OrdinalEncoder
from sklearn.metrics import silhouette_score
import sys
import matplotlib.pyplot as plt

def fetch_data():
    API_URL = "https://api.eia.gov/v2/petroleum/pri/fut/data/?api_key=xZfc9smFchx7pxAglwzNwLhXJTbtaYVGTijcoab1&frequency=weekly&data[0]=value&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
    response = requests.get(API_URL)
    data = response.json()
    df_raw = pd.DataFrame(data["response"]["data"])    
    print(df_raw)
    return df_raw

def bbl_to_gal(df):
    df['value'] = pd.to_numeric(df['value'],errors='coerce')
    units_filter = df['units']=='$/BBL'
    df.loc[units_filter, 'value'] *= 1/42
    df = df.drop('units', axis=1)
    return df

def period_to_year_month(df):
    df['period'] = pd.to_datetime(df['period'])
    df['year'] = df['period'].dt.year
    df['month'] = df['period'].dt.month
    del df['period']
    return df

def categorical_encoding(df):
    categorical_cols = df.select_dtypes(include=['object']).columns
    enc = OrdinalEncoder()
    df[categorical_cols]=enc.fit_transform(df[categorical_cols])
    return df

def spectral(df):
    X = df.values
    n_clusters = 9
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='rbf',random_state=51)
    labels = spectral.fit_predict(X)
    score = silhouette_score(df, labels)
    print(score)

def db(df):
    eps = 0.5
    min_samples = 3
    db = DBSCAN(eps = eps, min_samples=min_samples)
    labels = db.fit_predict(df)
    sil_score = silhouette_score(df, labels)
    print(sil_score, len(set(labels)))
    df['Cluster_Label'] = labels
    df.to_csv('NYMEX_futures_DBSCAN.csv',index=False)

def gmm(df):
    X = df.values
    n_components =  10
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X)
    labels = gmm.predict(X)
    loglikely = gmm.score(X) * X.shape[0]
    sil_score = silhouette_score(df, labels)
    print(loglikely)
    print(sil_score)


if __name__ == "__main__":
    df_raw = fetch_data()
    df_norm0 = bbl_to_gal(df_raw)
    df_norm1 = period_to_year_month(df_norm0)
    df_norm1.to_csv('NYMEX_futures.csv',index=False)
    df_norm2 = categorical_encoding(df_norm1)
    df_norm2.to_csv('NYMEX_futures_encoded.csv', index=False)

    if len(sys.argv) == 2:
        if sys.argv[1] == 'sc':
            spectral(df_norm2)
        elif sys.argv[1] == 'db':
            db(df_norm2)
        elif sys.argv[1] == 'gmm':
            gmm(df_norm2)
    else:
        print("no method selected")