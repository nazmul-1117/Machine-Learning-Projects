## **🚀 Workflow Overview**

```
RAW DATA → EDA → DATA PREPROCESSING → MODEL TRAINING → MODEL EVALUATION → SAVE MODEL → PREDICTIONS/DEPLOYMENT
```

---

### **1️⃣ Get the Data**

* Place the original Kaggle dataset (`creditcard.csv`) in:

  ```
  data/raw/
  ```
* This is your **immutable reference copy** — never edit it directly.

---

### **2️⃣ Exploratory Data Analysis (EDA)**

📍 Notebook: `notebooks/01-eda.ipynb`
Tasks:

* Load raw dataset.
* Understand data structure (columns, types, missing values).
* Visualize distributions & correlations.
* Look for **class imbalance** (fraud cases are usually <1%).
* Note down useful insights for preprocessing (e.g., need scaling? log-transform?).

**Outputs:**

* Understanding of necessary preprocessing steps.
* Initial baseline metrics (optional quick test model).

---

### **3️⃣ Data Preprocessing & Feature Engineering**

📍 Script: `src/preprocess_data.py`
Tasks:

* Handle missing values (if any).
* Scale numerical features (e.g., `StandardScaler` for PCA components).
* Split into **train/test** sets.
* Apply resampling if needed (e.g., SMOTE for imbalance).
* Save processed data to:

  ```
  data/processed/processed_data.csv
  ```

Run from command line:

```bash
python src/preprocess_data.py
```

---

### **4️⃣ Model Training**

📍 Notebook for experimentation: `notebooks/02-modeling.ipynb`
📍 Script for final model: `src/train_model.py`

Tasks:

* Try different algorithms (Logistic Regression, Random Forest, XGBoost, etc.).
* Tune hyperparameters (GridSearchCV or RandomizedSearchCV).
* Evaluate with metrics for imbalanced data:

  * Precision
  * Recall
  * F1-score
  * ROC-AUC
* Select **best performing model**.
* Save it to:

  ```
  models/fraud_model.pkl
  ```

Run final training:

```bash
python src/train_model.py
```

---

### **5️⃣ Predictions / Inference**

📍 Script: `src/predict.py`
Tasks:

* Load saved model.
* Accept new transaction data.
* Output probability or fraud prediction.

Example run:

```bash
python src/predict.py --input "sample_transaction.json"
```

---

### **6️⃣ Deployment (Optional)**

* Wrap prediction script into:

  * A **Flask / FastAPI API**.
  * A simple **Streamlit web app** for interactive use.
* Containerize with Docker for easy deployment.

---

### **🔄 Suggested Execution Order**

1. `notebooks/01-eda.ipynb` → Explore data, decide preprocessing steps.
2. `python src/preprocess_data.py` → Clean & prepare data.
3. `notebooks/02-modeling.ipynb` → Experiment with models.
4. `python src/train_model.py` → Train & save final model.
5. `python src/predict.py` → Test predictions.
6. (Optional) Build API/UI → Deploy model.

---