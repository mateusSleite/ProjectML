from flask import Flask, jsonify, request
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pandas as pd

app = Flask(__name__)
df = pd.read_csv('alg\\creditcard.csv')

def preprocess_data(df):
    df.fillna(df.mean(), inplace=True)
    df.drop_duplicates(inplace=True)

    scaler = StandardScaler()
    df[['V1', 'V2', 'V3', 'Amount']] = scaler.fit_transform(df[['V1', 'V2', 'V3', 'Amount']])

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df[['V1', 'V2', 'V3', 'Amount']])

    oversampler = RandomOverSampler()
    X_resampled, y_resampled = oversampler.fit_resample(df[['V1', 'V2', 'V3', 'Amount']], df['Class'])

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = preprocess_data(df)

def train_decision_tree():
    params = {'criterion': ['gini'],
              'max_depth': [None],
              'min_samples_split': [2],
              'min_samples_leaf': [1]}
    
    clf = GridSearchCV(DecisionTreeClassifier(random_state=42), params, cv=5, scoring='accuracy')
    clf.fit(X_train, y_train)
    
    best_clf = clf.best_estimator_
    accuracy, report, cm = evaluate_model(best_clf, X_test, y_test)
    results = {
        "best_params": clf.best_params_,
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }

    return jsonify(results)

def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, report, cm

@app.route('/train_decision_tree', methods=['GET'])
def train_decision_tree_route():
    return train_decision_tree()

if __name__ == '__main__':
    app.run(debug=True)


# from flask import Flask, jsonify, request
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from imblearn.over_sampling import RandomOverSampler
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
# import pandas as pd

# app = Flask(__name__)

# def preprocess_data(df):
#     df.fillna(df.mean(), inplace=True)
#     df.drop_duplicates(inplace=True)

#     scaler = StandardScaler()
#     df[['V1', 'V2', 'V3', 'Amount']] = scaler.fit_transform(df[['V1', 'V2', 'V3', 'Amount']])

#     pca = PCA(n_components=2)
#     principal_components = pca.fit_transform(df[['V1', 'V2', 'V3', 'Amount']])

#     oversampler = RandomOverSampler()
#     X_resampled, y_resampled = oversampler.fit_resample(df[['V1', 'V2', 'V3', 'Amount']], df['Class'])

#     X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

#     return X_train, X_test, y_train, y_test

# def train_decision_tree(X_train, y_train):
#     params = {'criterion': ['gini', 'entropy'],
#               'max_depth': [None, 5, 10, 15, 20],
#               'min_samples_split': [2, 5, 10],
#               'min_samples_leaf': [1, 2, 4]}
    
#     clf = GridSearchCV(DecisionTreeClassifier(random_state=42), params, cv=5, scoring='accuracy')
#     clf.fit(X_train, y_train)
#     return clf

# def train_logistic_regression(X_train, y_train):
#     params = {'penalty': ['l1', 'l2'],
#               'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    
#     clf = GridSearchCV(LogisticRegression(random_state=42), params, cv=5, scoring='accuracy')
#     clf.fit(X_train, y_train)
#     return clf

# def train_random_forest(X_train, y_train):
#     params = {'n_estimators': [50, 100, 200],
#               'max_depth': [None, 5, 10, 15, 20],
#               'min_samples_split': [2, 5, 10],
#               'min_samples_leaf': [1, 2, 4]}
    
#     clf = GridSearchCV(RandomForestClassifier(random_state=42), params, cv=5, scoring='accuracy')
#     clf.fit(X_train, y_train)
#     return clf

# def train_svm(X_train, y_train):
#     params = {'C': [0.1, 1, 10, 100],
#               'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
    
#     clf = GridSearchCV(SVC(random_state=42), params, cv=5, scoring='accuracy')
#     clf.fit(X_train, y_train)
#     return clf

# def train_bagging(X_train, y_train):
#     params = {'n_estimators': [10, 50, 100]}
    
#     clf = GridSearchCV(BaggingClassifier(random_state=42), params, cv=5, scoring='accuracy')
#     clf.fit(X_train, y_train)
#     return clf

# def evaluate_model(clf, X_test, y_test):
#     y_pred = clf.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     report = classification_report(y_test, y_pred)
#     cm = confusion_matrix(y_test, y_pred)
#     return accuracy, report, cm

# @app.route('/train_models', methods=['GET'])
# def train_models():
#     model_name = request.args.get('model')
#     df = pd.read_csv('alg\\creditcard.csv')
#     X_train, X_test, y_train, y_test = preprocess_data(df)

#     if model_name == 'Decision Tree':
#         clf = train_decision_tree(X_train, y_train)
#     elif model_name == 'Logistic Regression':
#         clf = train_logistic_regression(X_train, y_train)
#     elif model_name == 'Random Forest':
#         clf = train_random_forest(X_train, y_train)
#     elif model_name == 'SVM':
#         clf = train_svm(X_train, y_train)
#     elif model_name == 'Bagging':
#         clf = train_bagging(X_train, y_train)
#     else:
#         return jsonify({'error':'Invalid model name'})

#     best_clf = clf.best_estimator_
#     accuracy, report, cm = evaluate_model(best_clf, X_test, y_test)
#     results = {
#         "best_params": clf.best_params_,
#         "accuracy": accuracy,
#         "classification_report": report,
#         "confusion_matrix": cm.tolist()
#     }

#     return jsonify(results)

# if __name__ == '__main__':
#     app.run(debug=True)
