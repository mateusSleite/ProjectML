from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

trained_models = {}

def preprocess_data(df):
    df.fillna(df.mean(), inplace=True)
    df.drop_duplicates(inplace=True)

    scaler = StandardScaler()
    df[['V1', 'V2', 'V3', 'Amount']] = scaler.fit_transform(df[['V1', 'V2', 'V3', 'Amount']])

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df[['V1', 'V2', 'V3', 'Amount']])

    oversampler = RandomOverSampler()
    X_resampled, y_resampled = oversampler.fit_resample(principal_components, df['Class'])

    X_train, _, y_train, _ = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    return X_train, y_train

def train_decision_tree(X_train, y_train):
    params = {'criterion': ['gini'],
              'max_depth': [None],
              'min_samples_split': [2],
              'min_samples_leaf': [1]}
    
    clf = GridSearchCV(DecisionTreeClassifier(random_state=42), params, cv=5, scoring='accuracy')
    clf.fit(X_train, y_train)
    return clf

def train_elastic_net(X_train, y_train):
    params = {'penalty': ['elasticnet'],
              'alpha': [0.0001],
              'l1_ratio': [0.1]}
    
    clf = GridSearchCV(SGDClassifier(loss='log_loss', random_state=42), params, cv=5, scoring='accuracy')
    clf.fit(X_train, y_train)
    return clf

def train_random_forest(X_train, y_train):
    params = {'n_estimators': [50],
              'max_depth': [None],
              'min_samples_split': [2],
              'min_samples_leaf': [1]}
    
    clf = GridSearchCV(RandomForestClassifier(random_state=42), params, cv=5, scoring='accuracy')
    clf.fit(X_train, y_train)
    return clf

def train_svm(X_train, y_train):
    params = {'C': [100],
              'kernel': ['rbf']}
    
    clf = GridSearchCV(SVC(random_state=42), params, cv=5, scoring='accuracy')
    clf.fit(X_train, y_train)
    return clf

def train_bagging(X_train, y_train):
    params = {'n_estimators': [10]}
    
    clf = GridSearchCV(BaggingClassifier(random_state=42), params, cv=5, scoring='accuracy')
    clf.fit(X_train, y_train)
    return clf

def train_models():
    df = pd.read_csv('alg\\creditcard.csv')
    X_train, y_train = preprocess_data(df)

    trained_models['Decision Tree'] = train_decision_tree(X_train, y_train)
    trained_models['Elastic Net'] = train_elastic_net(X_train, y_train)
    trained_models['Random Forest'] = train_random_forest(X_train, y_train)
    trained_models['SVM'] = train_svm(X_train, y_train)
    trained_models['Bagging'] = train_bagging(X_train, y_train)

def evaluate_model(model_name, X_test, y_test):
    if model_name not in trained_models:
        return {'error':'Model not found'}

    clf = trained_models[model_name]
    accuracy = clf.score(X_test, y_test)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    results = {
        "best_params": clf.best_params_,
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }

    return results
