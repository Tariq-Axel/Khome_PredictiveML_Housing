#!/usr/bin/env python3.8


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

#Test of the decision tree regressor
from sklearn.tree import DecisionTreeRegressor

# 2. Scatter plot

def scatter_df(df, y_var):
    scatter_df = df.drop(y_var, axis = 1)
    dfcolumns = df.columns
    
    for counter in range(1,4,1):
        plot2 = sb.scatterplot(dfcolumns[counter], y_var, data = df, color = 'yellow', edgecolor = 'b', s = 150)
        plt.title('{} / y_var'.format(dfcolumns[counter]), fontsize = 16)
        plt.xlabel('{}'.format(dfcolumns[counter]), fontsize = 14)
        plt.ylabel('y_var', fontsize = 14)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.savefig('scatter2.png')
        plt.show()
        
def salepricedistrib(df):
    plt.title('Sale Price Distribution', fontsize = 16)
    plt.xlabel('date_mutation', fontsize = 14)
    plt.ylabel('valeur_fonciere', fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.scatter(df['date_mutation'], df['valeur_fonciere'], c = 'red')
    plt.savefig('distplot.png')
    plt.show()
    
def modeling(df):
    # FEATURE SELECTION & DATA SPLIT
    df["date_mutation"] = df.date_mutation.values.astype(np.int64) // 10 ** 9
    X_var = df[['date_mutation', 'code_postal','surface_reelle_bati','nombre_pieces_principales', 'latitude', 'longitude', 'adresse_code_voie']].values
    y_var = df['valeur_fonciere'].values

    X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size = 0.2, random_state = 0)

    print(cl('X_train samples : ', attrs = ['bold']), X_train[0:5])
    print(cl('X_test samples : ', attrs = ['bold']), X_test[0:5])
    print(cl('y_train samples : ', attrs = ['bold']), y_train[0:5])
    print(cl('y_test samples : ', attrs = ['bold']), y_test[0:5])


    # MODELING

    # 1. OLS

    ols = LinearRegression()
    ols.fit(X_train, y_train)
    ols_yhat = ols.predict(X_test)

    # 2. Ridge

    ridge = Ridge(alpha = 0.5)
    ridge.fit(X_train, y_train)
    ridge_yhat = ridge.predict(X_test)

    # 3. Lasso

    lasso = Lasso(alpha = 0.01)
    lasso.fit(X_train, y_train)
    lasso_yhat = lasso.predict(X_test)

    # 4. Bayesian

    bayesian = BayesianRidge()
    bayesian.fit(X_train, y_train)
    bayesian_yhat = bayesian.predict(X_test)

    # 5. ElasticNet

    en = ElasticNet(alpha = 0.01)
    en.fit(X_train, y_train)
    en_yhat = en.predict(X_test)


    ##Results
    # 1. Explained Variance Score

    print(cl('EXPLAINED VARIANCE SCORE:', attrs = ['bold']))
    print('-------------------------------------------------------------------------------')
    print(cl('Explained Variance Score of OLS model is {}'.format(evs(y_test, ols_yhat)), attrs = ['bold']))
    print('-------------------------------------------------------------------------------')
    print(cl('Explained Variance Score of Ridge model is {}'.format(evs(y_test, ridge_yhat)), attrs = ['bold']))
    print('-------------------------------------------------------------------------------')
    print(cl('Explained Variance Score of Lasso model is {}'.format(evs(y_test, lasso_yhat)), attrs = ['bold']))
    print('-------------------------------------------------------------------------------')
    print(cl('Explained Variance Score of Bayesian model is {}'.format(evs(y_test, bayesian_yhat)), attrs = ['bold']))
    print('-------------------------------------------------------------------------------')
    print(cl('Explained Variance Score of ElasticNet is {}'.format(evs(y_test, en_yhat)), attrs = ['bold']))
    print('-------------------------------------------------------------------------------')


    # 2. R-squared

    print(cl('R-SQUARED:', attrs = ['bold']))
    print('-------------------------------------------------------------------------------')
    print(cl('R-Squared of OLS model is {}'.format(r2(y_test, ols_yhat)), attrs = ['bold']))
    print('-------------------------------------------------------------------------------')
    print(cl('R-Squared of Ridge model is {}'.format(r2(y_test, ridge_yhat)), attrs = ['bold']))
    print('-------------------------------------------------------------------------------')
    print(cl('R-Squared of Lasso model is {}'.format(r2(y_test, lasso_yhat)), attrs = ['bold']))
    print('-------------------------------------------------------------------------------')
    print(cl('R-Squared of Bayesian model is {}'.format(r2(y_test, bayesian_yhat)), attrs = ['bold']))
    print('-------------------------------------------------------------------------------')
    print(cl('R-Squared of ElasticNet is {}'.format(r2(y_test, en_yhat)), attrs = ['bold']))
    print('-------------------------------------------------------------------------------')
