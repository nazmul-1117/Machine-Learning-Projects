## 📂 Project Structure

```
003-Credit_Card_Fraud_Detection/
│
├── data/
│   ├── raw/
│   │   └── creditcard.csv          # Original, unmodified Kaggle dataset
│   │
│   ├── processed/
│   │   └── processed_data.csv      # Cleaned, feature-engineered data ready for modeling
│
├── notebooks/
│   ├── 01-eda.ipynb                # Exploratory Data Analysis (EDA)
│   ├── 02-modeling.ipynb           # Model training, testing, and evaluation
│
├── src/
│   ├── __init__.py                  # Makes `src` a package
│   ├── preprocess_data.py           # Data cleaning & feature engineering script
│   ├── train_model.py               # Script to train and save the final model
│   ├── predict.py                    # Script to load model & make predictions
│
├── models/
│   └── fraud_model.pkl              # Serialized trained model
│
├── README.md                        # Project overview, setup, and usage instructions
├── requirements.txt                 # Python dependencies with versions
├── .gitignore                       # Ignore large data files, models, cache files, etc.
```

---

**💡 Explanation of Each Folder & File**

1. **`data/`**

   * **raw/**: Stores untouched datasets (always keep them unchanged for reference).
   * **processed/**: Contains cleaned datasets after feature engineering, scaling, and handling missing values.

2. **`notebooks/`**

   * Use Jupyter notebooks for experiments, visualizations, and documenting thought processes.
   * Separate notebooks keep EDA and modeling organized.

3. **`src/`**

   * All **production-ready code** goes here.
   * Scripts are modular so you can run them independently (e.g., preprocess without retraining).

4. **`models/`**

   * Save the final trained ML model here so you can quickly reload it for predictions without retraining.

5. **`README.md`**

   * Should explain:

     * Project goal
     * Dataset source
     * Steps to run the project
     * Key results
     * Any limitations or next steps

6. **`requirements.txt`**

   * Ensures reproducibility by listing exact dependencies, e.g.:

     ```
     pandas==2.0.3
     scikit-learn==1.3.0
     matplotlib==3.7.1
     seaborn==0.12.2
     ```

7. **`.gitignore`**

   * Prevents large/unnecessary files from being committed to version control, e.g.:

     ```
     data/raw/*
     models/*
     __pycache__/
     .ipynb_checkpoints/
     ```

---
