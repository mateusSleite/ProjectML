from flask import Flask, jsonify, request
from classification_methods import train_models, evaluate_model as evaluate_classification_model
from regression_models import train_regressors, evaluate_regressor as evaluate_regression_model

app = Flask(__name__)
train_models()
train_regressors()

@app.route('/evaluate_classification', methods=['POST'])
def evaluate_classification():
    model_name = request.json['model']
    X_test = request.json['X_test']
    y_test = request.json['y_test']

    results = evaluate_classification_model(model_name, X_test, y_test)
    return jsonify(results)

@app.route('/evaluate_regressor', methods=['POST'])
def evaluate_regressor():
    regressor_name = request.json['regressor']
    X_test = request.json['X_test']
    y_test = request.json['y_test']

    results = evaluate_regression_model(regressor_name, X_test, y_test)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
