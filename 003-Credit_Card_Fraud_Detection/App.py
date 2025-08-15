from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import os

import sys
import src.cyclical_transformer as cyclical_transformer
import src.drop_columns as drop_columns

sys.modules['cyclical_transformer'] = cyclical_transformer
sys.modules['drop_columns'] = drop_columns


app = Flask(__name__)

model_path = os.path.join('models', 'CCFD_XGB_Model.pkl')
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"✅ Model loaded from: {model_path}")
except FileNotFoundError:
    print(f"❌ Model file not found at: {model_path}")
except Exception as e:
    print(f"❌ Error loading model: {e}")


# Feature list
features = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

@app.route('/')
def home():
    return render_template('index.html', features=features)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input values into a dictionary
        input_data = {}
        for feature in features:
            val = request.form.get(feature)
            input_data[feature] = float(val)

        print('Input data: XD')
        # Convert to DataFrame with one row
        input_df = pd.DataFrame([input_data])
        print('Input DataFrame:', input_df)

        # Make prediction
        prediction = model.predict(input_df)[0]

        # Interpret result
        result = "Fraudulent Transaction 🚨" if prediction == 1 else "Legitimate Transaction ✅"
        return render_template('index.html', features=features, prediction=result)

    except Exception as e:
        print(f"❌ Error during prediction: ---> \n")
        return render_template('index.html', features=features, prediction=f"Error during prediction: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)
