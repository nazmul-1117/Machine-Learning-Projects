# Telecom Customer Churn Prediction Project Documentation

![Churn-Images](assets/chrun1.png)

## 1. Project Title

**TELECOM CUSTOMER CHURN PREDICTION 📈**

## 2. Introduction

Customer churn is a critical issue for businesses, especially in competitive sectors like telecommunications. This project focuses on predicting which customers are likely to discontinue their service or product before they actually do so. By identifying at-risk customers early, businesses can proactively implement retention strategies, which are often more cost-effective than acquiring new customers.

### 2.1 What is Customer Churn?

Customer churn refers to the phenomenon where customers stop doing business with a company or service. It signifies a loss of customers and, consequently, revenue.

### 2.2 What is Churn Prediction?

Churn prediction involves using data analysis and machine learning models to:

* **Identify At-Risk Customers:** Analyze various customer data points to determine the likelihood of them leaving.

* **Act as an Early Warning System:** Alert businesses to potential customer loss before it occurs.

* **Employ a Data-Driven Approach:** Rely on analyzing customer behavior, transaction history, engagement data, and other relevant information to identify churn patterns.

* **Facilitate Preventative Measures:** Enable targeted interventions like personalized offers, improved customer support, or tailored communication to prevent churn.

### 2.3 Why is Churn Prediction Important?

* **Cost Savings:** Retaining existing customers is generally less expensive than acquiring new ones.

* **Revenue Retention:** Preventing churn helps maintain and stabilize revenue streams.

* **Improved Customer Lifetime Value (CLTV):** Longer customer retention increases the overall value derived from each customer.

* **Competitive Advantage:** Proactive churn prediction and prevention enhance a business's competitive edge by fostering a stable and loyal customer base.

### 2.4 How is Churn Predicted?

The process typically involves:

1.  **Data Collection:** Gathering relevant customer behavior data (e.g., purchase history, website activity, support interactions, demographics).

2.  **Model Selection:** Choosing appropriate machine learning models (e.g., Logistic Regression, Decision Trees, Ensemble Methods).

3.  **Model Training and Evaluation:** Training the selected model on historical data and rigorously evaluating its performance for accuracy.

4.  **Prediction and Action:** Using the trained model to predict at-risk customers and implementing targeted retention strategies.

### 2.5 Examples of Churn Prediction in Different Industries

* **Subscription Services:** Identifying customers with reduced usage or inactivity.

* **E-commerce:** Detecting customers who haven't made recent purchases or have abandoned carts.

* **Telecommunications:** Analyzing usage patterns, billing issues, or customer service interactions.

### 2.6 Business Objectives

The primary business objectives of this project are to:

* **Retain Customers:** Implement strategies to keep existing customers.

* **Identify At-Risk Customers Early:** Enable timely interventions (discounts, promotions, personalized communication).

* **Increase Revenue and Loyalty:** Improve the precision of retention efforts.

* **Analyze Churn Patterns:**

    * Determine the percentage of churned vs. active customers.

    * Identify churn patterns based on gender.

    * Uncover patterns/preferences in churn based on service types.

    * Identify the most profitable service types and features.

* Address other questions that may arise during the analysis.

## 3. Project Workflow

The project follows a standard machine learning workflow:

1.  **Import Libraries:** Essential Python libraries for data manipulation, visualization, preprocessing, and machine learning.

2.  **Load Dataset:** Loading the Telco Customer Churn dataset.

3.  **Understanding The Data:** Initial exploration of the dataset's structure, columns, and basic statistics.

4.  **Data Visualization:** Visualizing key features to identify patterns and insights related to churn.

5.  **Data Preprocessing:**

    * Data Cleaning

    * Data Transformation

    * Data Scaling

    * Data Integration

    * Data Reduction

6.  **Split Data:** Dividing the dataset into training, validation, and test sets, including handling imbalanced datasets.

7.  **Machine Learning Model Evaluations and Predictions:** Training and evaluating various machine learning models.

8.  **Improve the Model:** Iterative refinement of the selected model.

9.  **Save Model:** Persisting the trained model for future use.

10. **References:** Acknowledging sources and resources used.

## 4. Dataset Overview

The dataset contains information about Telco customers, including:

* **Customers who left within the last month** (target variable: `Churn`).

* **Services each customer has signed up for:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`.

* **Customer account information:** `tenure` (how long they’ve been a customer), `Contract`, `PaymentMethod`, `PaperlessBilling`, `MonthlyCharges`, `TotalCharges`.

* **Demographic info:** `gender`, `SeniorCitizen` (age range), `Partner`, `Dependents`.

### 4.1 Column Descriptions

| Column          | Type    | Description                                       |
| :-------------- | :------ | :------------------------------------------------ |
| `customerID`    | object  | Unique customer identifier (can be dropped for ML) |
| `gender`        | object  | Male/Female                                       |
| `SeniorCitizen` | int64   | 1 if senior, 0 otherwise                          |
| `Partner`       | object  | Yes/No – has a partner                            |
| `Dependents`    | object  | Yes/No – has dependents                           |
| `tenure`        | int64   | Number of months the customer has stayed          |
| `PhoneService`  | object  | Yes/No                                            |
| `MultipleLines` | object  | No, Yes, or No phone service                      |
| `InternetService` | object | DSL, Fiber optic, or No                           |
| `OnlineSecurity` | object | Yes/No/No internet service                        |
| `OnlineBackup`  | object  | Same as above                                     |
| `DeviceProtection` | object | Same as above                                     |
| `TechSupport`   | object  | Same as above                                     |
| `StreamingTV`   | object  | Same as above                                     |
| `StreamingMovies` | object | Same as above                                     |
| `Contract`      | object  | Month-to-month, One year, Two year                |
| `PaperlessBilling` | object | Yes/No                                            |
| `PaymentMethod` | object  | 4 methods (e.g., Electronic check, Mailed check)  |
| `MonthlyCharges` | float64 | The amount charged to the customer monthly        |
| `TotalCharges`  | object ⛔️ | Needs conversion to numeric (contains invalid/missing entries) |
| `Churn`         | object ✅ | **Target** variable: Yes = churned, No = stayed   |

### 4.2 Feature Categorization

* **Drop Columns:** `customerID`

* **Numerical Features:** `tenure`, `MonthlyCharges`, `TotalCharges`

* **Categorical Features:**

    * **Binary:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `PaperlessBilling`

    * **Nominal:** `InternetService`, `PaymentMethod`

    * **Ordinal:** `Contract`

* **Alphanumeric Features:** None

* **Missing Values:** No missing values identified initially (however, `TotalCharges` needs handling for empty strings).

* **Output Column:** `Churn`

## 5. Key Data Visualizations and Insights

### 5.1 Churned Customers – Gender Ratio

* **50.2%** of churned customers are Female.

* **49.8%** of churned customers are Male.

* **Insight:** Gender does not appear to be a significant factor in churn, as the distribution is almost equal.

### 5.2 Churned Customers – Senior Citizen Ratio

* **25.5%** of churned customers are Senior Citizens.

* **74.5%** of churned customers are Non-Senior Citizens.

* **Insight:** Non-senior citizens constitute a larger portion of churned customers, suggesting that age might play a role in churn behavior, with younger/non-senior demographics being more prone to leaving.

### 5.3 Tenure Grouping

Customers are grouped into tenure categories (e.g., `<1yr`, `1st_Year`, `2nd_Year`, etc.) to analyze churn based on how long they have been a customer. This helps identify if churn is more prevalent in early or later stages of a customer's lifecycle.

## 6. Machine Learning Models and Performance

The project evaluates various machine learning models for churn prediction, with **Stacking** emerging as the best-performing model.

### 6.1 Model Accuracy Comparison

| Model Name          | Accuracy (%) |
| :------------------ | :----------- |
| Stacking            | 82.42        |
| Logistic Regression | 82.17        |
| Gradient Boosting   | 81.98        |
| XGBoost             | 81.21        |
| MLPClassifier       | 81.01        |
| KNN                 | 79.81        |
| Random Forest       | 79.42        |
| Decision Tree       | 78.02        |
| Naive Bayes         | 77.73        |

### 6.2 Final Model Performance: Stacking

| Dataset    | Accuracy % |
| :--------- | :--------- |
| Train      | 94.97      |
| Validation | 84.35      |
| Test       | 84.98      |

* The **Stacking model** achieved a robust **84.98% accuracy on the unseen test data**.

* While there was a notable gap between training (94.97%) and test (84.98%) accuracy, indicating some overfitting, the strong generalization performance on new data confirms the model's reliability for predicting churn.

## 7. Data Preprocessing Details (Conceptual)

The notebook outlines several preprocessing steps that are crucial for preparing the data for machine learning:

* **Data Cleaning:** Handling missing values (specifically converting `TotalCharges` to numeric and addressing empty strings).

* **Data Transformation:** Encoding categorical variables (binary, nominal, ordinal) into numerical representations suitable for machine learning algorithms.

* **Data Scaling:** Scaling numerical features to ensure they contribute equally to the model training process, preventing features with larger values from dominating.

* **Data Integration:** Combining data from various sources if necessary (though not explicitly detailed in the provided snippet, it's a general step).

* **Data Reduction:** Reducing the number of features if needed to improve model performance and reduce complexity.

## 8. Libraries Used

* **Data Manipulation:** `pandas`, `numpy`

* **Data Visualization:** `matplotlib.pyplot`, `seaborn`

* **Preprocessing:** `sklearn.preprocessing` (OrdinalEncoder, OneHotEncoder, LabelEncoder, StandardScaler)

* **Imbalanced Dataset Handling:** `imblearn.over_sampling` (SMOTE)

* **Model Selection:** `sklearn.model_selection` (train_test_split, cross_val_score, cross_validate, GridSearchCV, RandomizedSearchCV)

* **Machine Learning Models:** `sklearn.linear_model` (LogisticRegression), `sklearn.tree` (DecisionTreeClassifier), `sklearn.svm` (SVC), `sklearn.neighbors` (KNeighborsClassifier), `sklearn.naive_bayes` (BernoulliNB), `sklearn.ensemble` (StackingClassifier, RandomForestClassifier, GradientBoostingClassifier), `sklearn.neural_network` (MLPClassifier), `xgboost` (XGBClassifier)

* **Metrics:** `sklearn.metrics` (accuracy_score, confusion_matrix, f1_score, recall_score, ConfusionMatrixDisplay, classification_report, roc_auc_score, roc_curve)

* **Persistence:** `pickle`, `os`



## 9\. Model Building and Evaluation

We evaluated a range of classification algorithms to identify the most effective model for this task.

### Models Tested:

  * Logistic Regression

  * K-Nearest Neighbors (KNN)

  * Naive Bayes

  * Decision Tree

  * Random Forest

  * SVM

  * Gradient Boosting

  * XGBoost

  * MLPClassifier (Neural Network)

  * **Stacking Classifier** (combining multiple models)

### Performance Comparison:

The models were evaluated based on their accuracy on a held-out test set. The Stacking Classifier emerged as the top-performing model.

| Model             | Accuracy (%) |
| :---------------- | :----------- |
| **Stacking** | **82.46%** |
| Logistic Regression | 82.17%       |
| Gradient Boosting | 81.88%       |
| XGBoost           | 81.21%       |
| MLPClassifier     | 80.00%       |
| KNN               | 79.81%       |
| Random Forest     | 79.52%       |
| Decision Tree     | 78.02%       |
| Naive Bayes       | 77.73%       |


### Final Model Performance: Stacking

The chosen model, **Stacking**, demonstrated robust performance across all data splits, confirming its ability to generalize well to new, unseen data.

| Dataset    | Accuracy |
| :--------- | :------- |
| Train      | 94.97%   |
| Validation | 84.35%   |
| **Test** | **84.98%** |

The high accuracy on the test set indicates that the model is reliable for predicting which customers are likely to churn.

## 10\. How to Use This Project

1.  **Clone the repository:**

    ```
    git clone https://github.com/nazmul-1117/Machine-Learning-Projects.git
    cd 002-Customer_Churn_Prediction
    ```

2.  **Install dependencies:**

    ```
    pip install -r requirements.txt
    ```

3.  **Run the analysis:**

      * Open and run the `Customer_Churn_Prediction.ipynb` notebook in a Jupyter environment to see the full analysis and model training process.

      * To make predictions with the pre-trained model, load `CUSTOMER_CHURN_PREDICTION.pkl` and use it on new data.

## 11\. Future Work

  * **Hyperparameter Tuning:** Further optimize the Stacking model's parameters using `GridSearchCV` or `RandomizedSearchCV`.

  * **Feature Engineering:** Create new features, such as `TenureInYears` or service usage ratios, to potentially improve model performance.

  * **Deployment:** Package the final model into a REST API using Flask or FastAPI for integration into a production environment.

## 12\. Download Model
Download the model from [Github](https://github.com/nazmul-1117/Machine-Learning-Projects/002-Customer_Churn_Prediction/saved_models/CUSTOMER_CHURN_PREDICTION.pkl).


## 13\. AUTHOR

**Author:** [Md. Nazmul Hossain](https://github.com/nazmul-1117)

**Kaggle:** [Md. Nazmul Hossain](https://www.kaggle.com/nazmul1117)

**Contact:** [Instagram](https://www.instagram.com/nazmul.1117/)