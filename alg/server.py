from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import os
import tempfile
from classification_methods import train_models, evaluate_model as evaluate_classification_model
from regression_models import train_regressors, evaluate_regressor as evaluate_regression_model

app = Flask(__name__)
CORS(app)

train_models()
train_regressors()

def preprocess_csv_for_evaluation(filepath, is_classification):
    df = pd.read_csv(filepath)
    scaler = StandardScaler()
    df[['V1', 'V2', 'V3', 'Amount']] = scaler.fit_transform(df[['V1', 'V2', 'V3', 'Amount']])
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df[['V1', 'V2', 'V3', 'Amount']])
    
    if is_classification:
        X_test = principal_components
        y_test = df['Class']
    else:
        X_test = principal_components
        y_test = df['Amount']

    return X_test, y_test

@app.route('/evaluate_classification', methods=['POST'])
def evaluate_classification():
    model_name = request.form['model']
    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, filename)
        file.save(filepath)
        
        X_test, y_test = preprocess_csv_for_evaluation(filepath, True)
        results = evaluate_classification_model(model_name, X_test, y_test)
        os.remove(filepath)
        print(results)
        return jsonify(results)
    else:
        return jsonify({'error': 'No file provided'}), 400

@app.route('/evaluate_regressor', methods=['POST'])
def evaluate_regressor():
    regressor_name = request.form['model']
    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, filename)
        file.save(filepath)
        
        X_test, y_test = preprocess_csv_for_evaluation(filepath, False)
        results = evaluate_regression_model(regressor_name, X_test, y_test)
        os.remove(filepath)
        
        return jsonify(results)
    else:
        return jsonify({'error': 'No file provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)
