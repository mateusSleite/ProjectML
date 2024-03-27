from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd

trained_regressors = {}

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

def train_regressors():
    df = pd.read_csv('alg\\creditcard.csv')
    X_train, _, y_train, _ = preprocess_data(df)

    trained_regressors['Decision Tree Regressor'] = train_decision_tree_regressor(X_train, y_train)
    trained_regressors['Elastic Net Regressor'] = train_elastic_net_regressor(X_train, y_train)
    trained_regressors['SVR'] = train_svr(X_train, y_train)
    trained_regressors['Random Forest Regressor'] = train_random_forest_regressor(X_train, y_train)
    trained_regressors['Bagging Regressor'] = train_bagging_regressor(X_train, y_train)

def evaluate_regressor(regressor_name, X_test, y_test):
    if regressor_name not in trained_regressors:
        return {'error':'Regressor not found'}

    reg = trained_regressors[regressor_name]
    y_pred = reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    results = {
        "best_params": reg.best_params_,
        "mean_squared_error": mse
    }

    return results
