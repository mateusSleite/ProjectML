from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

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

def train_decision_tree(X_train, y_train):
    params = {'criterion': ['gini', 'entropy'],
              'max_depth': [None, 5, 10, 15, 20],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4]}
    clf = GridSearchCV(DecisionTreeClassifier(random_state=42), params, cv=5, scoring='accuracy')
    clf.fit(X_train, y_train)
    print(clf.best_params_);
    return clf

def train_logistic_regression(X_train, y_train):
    params = {'penalty': ['l2'],
              'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    
    clf = GridSearchCV(LogisticRegression(random_state=42, solver='lbfgs'), params, cv=5, scoring='accuracy')
    clf.fit(X_train, y_train)
    print(clf.best_params_);
    return clf

def train_random_forest(X_train, y_train):
    params = {'n_estimators': [50, 100, 200],
              'max_depth': [None, 5, 10, 15, 20],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4]}
    
    clf = GridSearchCV(RandomForestClassifier(random_state=42), params, cv=5, scoring='accuracy')
    clf.fit(X_train, y_train)
    print(clf.best_params_);
    return clf

def train_svm(X_train, y_train):
    params = {'C': [0.1, 1, 10, 100],
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
    
    clf = GridSearchCV(SVC(random_state=42), params, cv=5, scoring='accuracy')
    clf.fit(X_train, y_train)
    print(clf.best_params_);
    return clf

def train_bagging(X_train, y_train):
    params = {'n_estimators': [10, 50, 100]}
    
    clf = GridSearchCV(BaggingClassifier(random_state=42), params, cv=5, scoring='accuracy')
    clf.fit(X_train, y_train)
    print(clf.best_params_);
    return clf

df = pd.read_csv('alg\\creditcard.csv')
X_train, X_test, y_train, y_test = preprocess_data(df)

# clf_decision_tree = train_decision_tree(X_train, y_train)
# clf_logistic_regression = train_logistic_regression(X_train, y_train)
# clf_random_forest = train_random_forest(X_train, y_train)
# clf_svm = train_svm(X_train, y_train)
clf_bagging = train_bagging(X_train, y_train)


# print("Decision Tree:")
# print(classification_report(y_test, clf_decision_tree.predict(X_test)))
# print(confusion_matrix(y_test, clf_decision_tree.predict(X_test)))

# print("\nLogistic Regression:")
# print(classification_report(y_test, clf_logistic_regression.predict(X_test)))
# print(confusion_matrix(y_test, clf_logistic_regression.predict(X_test)))

# print("\nRandom Forest:")
# print(classification_report(y_test, clf_random_forest.predict(X_test)))
# print(confusion_matrix(y_test, clf_random_forest.predict(X_test)))

# print("\nSVM:")
# print(classification_report(y_test, clf_svm.predict(X_test)))
# print(confusion_matrix(y_test, clf_svm.predict(X_test)))

print("\nBagging:")
print(classification_report(y_test, clf_bagging.predict(X_test)))
print(confusion_matrix(y_test, clf_bagging.predict(X_test)))
