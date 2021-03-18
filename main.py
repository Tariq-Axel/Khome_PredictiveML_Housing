#!/usr/bin/env python3.8
#https://medium.com/codex/house-price-prediction-with-machine-learning-in-python-cf9df744f7ff

from function import *

# IMPORTING PACKAGES
import pandas as pd # data processing
import numpy as np # working with arrays
import matplotlib.pyplot as plt # visualization
import seaborn as sb # visualization
from termcolor import colored as cl # text customization
from sklearn.model_selection import train_test_split # data split
from sklearn.linear_model import LinearRegression # OLS algorithm
from sklearn.linear_model import Ridge # Ridge algorithm
from sklearn.linear_model import Lasso # Lasso algorithm
from sklearn.linear_model import BayesianRidge # Bayesian algorithm
from sklearn.linear_model import ElasticNet # ElasticNet algorithm
from sklearn.metrics import explained_variance_score as evs # evaluation metric
from sklearn.metrics import r2_score as r2 # evaluation metric
import datetime as dt
import glob

#Test of the decision tree regressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree



sb.set_style('whitegrid') # plot style
plt.rcParams['figure.figsize'] = (20, 10) # plot size

#Getting Data from all files
all_files = glob.glob('../Datasets/*/*.csv')
li = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)
frame = pd.concat(li, axis=0, ignore_index=True)

frame=frame[(frame.type_local == 'Appartement')]
frame=frame[['id_mutation', 'date_mutation', 'valeur_fonciere', 'code_postal' , 'surface_reelle_bati','nombre_pieces_principales', 'latitude', 'longitude']]
frame=frame.dropna(axis=0)
#frame=frame.head(100)
frame.to_csv('frame.csv') # On enregistre la data

#On met en place la data de test
testdata = frame.iloc[:10, :] # On récupère les 10 premières lignes pour tester le modèle prédictif
print(testdata.head(10)) # On affiche les réponses
testdata.drop('valeur_fonciere', inplace=True, axis=1) # on enlève les valeurs foncières pour pouvoir tester après
testdata["date_mutation"] = pd.to_datetime(testdata["date_mutation"])#On converti en date
testdata["date_mutation"] = (testdata["date_mutation"]-testdata["date_mutation"].min())/ np.timedelta64(1,'D') #On calcule un nombre de jour depuis le min pour avoir un float
#print(testdata["adresse_code_voie"]) 'nom_commune',
testdata.set_index('id_mutation', inplace = True)

#On met en place la data pour l'entrainement
frame = frame.iloc[11:, :] # on laisse le reste pour l'entrainement

Y = frame.valeur_fonciere
X = frame
X.drop('valeur_fonciere', inplace=True, axis=1)
X["date_mutation"] = pd.to_datetime(X["date_mutation"])#On converti en date
X["date_mutation"] = (X["date_mutation"]-X["date_mutation"].min())/ np.timedelta64(1,'D') #On calcule un nombre de jour depuis le min pour avoir un float
#print(X["adresse_code_voie"]) 'nom_commune',
X.set_index('id_mutation', inplace = True)
#X['latitude'].replace('', np.nan, inplace=True)

X.to_csv('X.csv')

#print(f"{X.head(10)}") #display a few rows

#X.dropna(inplace = True)
#.dropna(axis=0)
X.fillna(X.mean())
print("test")
#print(cl(X.isnull().sum(), attrs = ['bold']))

#print(f"number of lines and columns : {X.shape}") #Number of lines and columns
print(f"number of lines : {len(X.index)}") #Number of lines (efficient way)

#pd.set_option('float_format', '{:f}'.format) #Option to display X.describe in float format
#print(f"{X.describe()}") #see count,mean, std, min etc...

#Heatmap to see the correlation between variables
    #X = X[X.nombre_pieces_principales.isin([2])]
    #sb.heatmap(X.corr(), annot = True, cmap = 'magma')
    #plt.savefig('heatmap.png')
    #plt.show()

#scatter_df(X, 'valeur_fonciere')#using a funtion

# 3. Distribution plot

#filter
#X = X[(X.code_postal == 94130)  & (X.nombre_pieces_principales == 3)]
#X = X[X.nombre_pieces_principales.isin(["2", "3"])]
#salepricedistrib(X)


#train dataset and check result 
#modeling(X)


# TEEEEEEEST
# Define model. Specify a number for random_state to ensure same results each run
predictive_model = DecisionTreeRegressor(random_state=1)

# Fit model
predictive_model.fit(X, Y)

print("Rappel des éléments à prédire :")
print(testdata.head(10))
print("Prédictions")
print(predictive_model.predict(testdata.head(10)))



text_representation = tree.export_text(predictive_model)
print(text_representation)


fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(predictive_model, 
                   feature_names=X.feature_names,  
                   class_names=Y.target_names,
                   filled=True)

fig.savefig("decistion_tree.png")
