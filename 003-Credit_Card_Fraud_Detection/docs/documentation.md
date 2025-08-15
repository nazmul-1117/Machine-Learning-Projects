# 🧾 Credit Card Fraud Detection – Flask App Documentation

## 📌 Project Overview

This project is a machine learning-powered web application designed to detect fraudulent credit card transactions. It uses an XGBoost model trained on preprocessed transaction data and is deployed via a Flask web interface.

---

## 🏗️ Project Structure

```
003-Credit_Card_Fraud_Detection/
├── App.py                         # Flask application
├── models/
│   └── CCFD_XGB_Model.pkl         # Trained XGBoost model (pickled)
├── src/
│   ├── __init__.py                # Package initializer
│   ├── cyclical_transformer.py    # Custom transformer for time features
│   └── drop_columns.py            # Utility to drop irrelevant columns
├── templates/
│   └── index.html                 # Frontend HTML form
├── static/                        # (Optional) CSS/JS assets
└── README.md                      # Project documentation
```

---

## ⚙️ Setup Instructions

#### 1. **Install Dependencies**

```bash
pip install flask xgboost scikit-learn pandas numpy
```

> Optional: Use a virtual environment to isolate dependencies.

#### 2. **Run the Flask App**

```bash
python App.py
```

Then open your browser to:  
**http://127.0.0.1:5000**

---

## 🧠 Model Details

- **Algorithm**: XGBoost Classifier
- **Accuracy**: 100% on training, validation, and unseen test data
- **Preprocessing**:
  - `CyclicalTimeTransformer`: Converts `Time` into sine/cosine features
  - `ColumnDropper`: Removes `Time` and `Amount_Category`
  - `StandardScaler`: Scales `Amount` feature
- **Pipeline**: All preprocessing and model steps are wrapped in a `Pipeline` object and saved as `CCFD_XGB_Model.pkl`

---

## 🧪 How Prediction Works

1. User inputs transaction features via the web form.
2. Flask app collects the values and converts them to a NumPy array.
3. The model pipeline transforms the input and makes a prediction.
4. The result is displayed as either **"Legitimate"** or **"Fraudulent"**.

---

## 🧰 Custom Transformers

#### `CyclicalTimeTransformer`

```python
class CyclicalTimeTransformer(BaseEstimator, TransformerMixin):
    def transform(self, X):
        time_mod = X['Time'] % 86400
        X['Time_sin'] = np.sin(2 * np.pi * time_mod / 86400)
        X['Time_cos'] = np.cos(2 * np.pi * time_mod / 86400)
        return X.drop(columns=['Time'])
```

#### `ColumnDropper`

```python
class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop)
```

---

## 🧩 Troubleshooting

#### ❌ Error: `No module named 'cyclical_transformer'`

**Cause**: Pickle expects the original module path used during model training.

**Fixes**:
- Ensure `cyclical_transformer.py` is inside `src/`
- Use absolute imports: `from src.cyclical_transformer import CyclicalTimeTransformer`
- Add `src` to `sys.path` if needed
- Use `dill` instead of `pickle` for more flexible loading

---

## 📦 Packaging Tips

To reuse your custom transformers across projects:

1. Create a `setup.py` in the root directory
2. Install locally with `pip install -e .`
3. Use `__init__.py` to expose key classes

---

## 🧠 Final Notes

- This app is ideal for demonstrating fraud detection logic.
- For production, consider adding:
  - Input validation
  - Logging
  - Model versioning
  - RESTful API endpoints
  - Docker containerization