from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def preprocess_data(df):
    df.fillna(df.mean(), inplace=True)
    df.drop_duplicates(inplace=True)

    scaler = StandardScaler()
    df[['V1', 'V2', 'V3', 'Amount']] = scaler.fit_transform(df[['V1', 'V2', 'V3', 'Amount']])

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df[['V1', 'V2', 'V3', 'Amount']])

    X_train, X_test, y_train, y_test = train_test_split(principal_components, df['Amount'], test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def train_decision_tree_regressor(X_train, y_train):
    params = {'criterion': ['squared_error'],
              'max_depth': [None],
              'min_samples_split': [2],
              'min_samples_leaf': [1]}
    
    reg = GridSearchCV(DecisionTreeRegressor(random_state=42), params, cv=5, scoring='neg_mean_squared_error')
    reg.fit(X_train, y_train)
    return reg

def train_elastic_net_regressor(X_train, y_train):
    params = {'alpha': [0.0001],
              'l1_ratio': [0.1]}
    
    reg = GridSearchCV(ElasticNet(random_state=42), params, cv=5, scoring='neg_mean_squared_error')
    reg.fit(X_train, y_train)
    return reg

def train_svr(X_train, y_train):
    params = {'C': [100],
              'kernel': ['rbf']}
    
    reg = GridSearchCV(SVR(), params, cv=5, scoring='neg_mean_squared_error')
    reg.fit(X_train, y_train)
    return reg

def train_random_forest_regressor(X_train, y_train):
    params = {'n_estimators': [50],
              'max_depth': [None],
              'min_samples_split': [2],
              'min_samples_leaf': [1]}
    
    reg = GridSearchCV(RandomForestRegressor(random_state=42), params, cv=5, scoring='neg_mean_squared_error')
    reg.fit(X_train, y_train)
    return reg

def train_bagging_regressor(X_train, y_train):
    params = {'n_estimators': [10]}
    
    reg = GridSearchCV(BaggingRegressor(random_state=42), params, cv=5, scoring='neg_mean_squared_error')
    reg.fit(X_train, y_train)
    return reg

def plot_results(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(y_test)), y_test, label='Valores Reais', marker='o', linestyle='-', color='b')
    plt.plot(np.arange(len(y_test)), y_pred, label='Valores Previstos', marker='x', linestyle='--', color='r')
    plt.title('Valores Reais vs Valores Previstos')
    plt.xlabel('Índice')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(True)
    plt.show()

df = pd.read_csv('alg\\creditcard.csv')
X_train, X_test, y_train, y_test = preprocess_data(df)

regressors = {
    'Decision Tree Regressor': train_decision_tree_regressor(X_train, y_train),
    'Elastic Net Regressor': train_elastic_net_regressor(X_train, y_train),
    'SVR Regressor': train_svr(X_train, y_train),
    'Random Forest Regressor': train_random_forest_regressor(X_train, y_train),
    'Bagging Regressor': train_bagging_regressor(X_train, y_train)
}

for name, regressor in regressors.items():
    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Regressor: {name}\nMean Squared Error: {mse}')
    plot_results(y_test, y_pred)
