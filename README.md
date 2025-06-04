## Credit Card Fraud Detection

This project implements and compares two machine learning approaches—**Logistic Regression** and **Isolation Forest**—for detecting credit card fraud in highly imbalanced datasets. The goal is to optimize fraud detection by minimizing both false positives and false negatives, which is critical in financial security.

### 🔍 Key Features

* 📊 **Exploratory Data Analysis (EDA):** In-depth analysis of credit card transaction data to understand distribution and identify patterns.
* 🤖 **Model Implementation:**

  * **Logistic Regression:** Includes class weight tuning to handle class imbalance.
  * **Isolation Forest:** Leverages the unsupervised nature of anomaly detection with tuned contamination rates.
* 🔧 **Hyperparameter Optimization:** Utilizes `GridSearchCV` for parameter tuning to improve model precision and recall.
* 🧮 **Custom Scoring Function:** Balances precision and recall to suit the specific needs of fraud detection.
* 📈 **Performance Visualization:** Visual comparison of model metrics across different hyperparameter values for interpretability.

### 🚀 Objectives

* Handle severely imbalanced data effectively.
* Optimize fraud detection algorithms with a strong focus on **precision-recall trade-offs**.
* Provide a reproducible and extensible baseline for financial anomaly detection projects.

