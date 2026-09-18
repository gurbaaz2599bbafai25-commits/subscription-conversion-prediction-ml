# Subscription Conversion Prediction — Machine Learning

An end-to-end machine learning project that predicts whether a free user of a subscription-based application will convert into a paid subscriber based on their engagement behavior.

## 📌 Project Overview

Subscription-based applications have a large number of free users, but only a portion of them eventually become paid customers.

The objective of this project is to build a machine learning classification system that can identify users who are more likely to convert to a paid subscription.

The predictions can help a business:

* Identify high-potential free users
* Prioritize conversion campaigns
* Personalize offers and messaging
* Improve marketing efficiency
* Understand engagement signals associated with conversion

## 🎯 Machine Learning Problem

This project is a **supervised binary classification problem**.

### Target Variable

`Converted`

* `1` → User converted to a paid subscriber
* `0` → User did not convert

## 📊 Dataset

The project uses a subscription conversion dataset containing user-level engagement information.

The provided raw CSV contains **119 records and 10 columns**. It includes duplicate user IDs and missing values, making data cleaning an important part of the project.

### Features

| Feature                | Description                                        |
| ---------------------- | -------------------------------------------------- |
| `User_ID`              | Unique user identifier                             |
| `Age`                  | User age                                           |
| `Days_Since_Signup`    | Number of days since registration                  |
| `Sessions`             | Number of sessions                                 |
| `Visits`               | Number of visits                                   |
| `Features_Used`        | Number of application features used                |
| `Avg_Session_Minutes`  | Average session duration                           |
| `Trial_Days_Used`      | Number of trial days used                          |
| `Support_Interactions` | Number of support interactions                     |
| `Converted`            | Target variable indicating subscription conversion |

## 🔄 Project Workflow

```text
Raw User Data
      ↓
Data Understanding
      ↓
Data Cleaning
      ↓
Outlier Detection & Treatment
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering
      ↓
Target Definition
      ↓
Feature Selection
      ↓
Target Encoding
      ↓
Train-Test Split
      ↓
Feature Standardisation
      ↓
Model Building
      ↓
Model Training
      ↓
Prediction
      ↓
Model Evaluation
      ↓
Best Model Selection
      ↓
Model Interpretation
      ↓
Prediction & Business Segmentation
```

## 🧹 Data Cleaning

The raw dataset is processed before model training.

The notebook performs:

* Duplicate-row removal
* Missing-value detection
* Median imputation for missing numerical values
* Outlier detection using the **IQR method**
* Outlier treatment through value capping/winsorization

Outliers are capped rather than removed because the dataset is relatively small and retaining user records is preferred.

## 📈 Exploratory Data Analysis

EDA is performed to understand:

* Dataset structure
* Data types
* Missing values
* Duplicate records
* Target-class distribution
* Summary statistics
* Relationship between engagement variables and subscription conversion

The project examines variables such as:

* Sessions
* Visits
* Features used
* Average session duration
* Trial days used
* Support interactions

## ⚙️ Feature Engineering

Two additional engagement-rate features are created:

### Sessions Per Day

```text
Sessions_Per_Day =
Sessions / Days_Since_Signup
```

### Visits Per Day

```text
Visits_Per_Day =
Visits / Days_Since_Signup
```

These features provide a normalized view of user engagement.

## 🔎 Feature Selection

`User_ID` is excluded from model training because it is an identifier rather than a predictive feature.

The model uses:

```text
Age
Days_Since_Signup
Sessions
Visits
Features_Used
Avg_Session_Minutes
Trial_Days_Used
Support_Interactions
Sessions_Per_Day
Visits_Per_Day
```

## 🤖 Machine Learning Models

Three classification algorithms are trained and compared:

### 1. Logistic Regression

Used as an interpretable linear classification baseline.

### 2. Decision Tree

Used to capture non-linear relationships and decision rules within the user engagement data.

### 3. Random Forest

An ensemble model consisting of multiple decision trees.

## 🧪 Train-Test Split

The dataset is divided into:

* **80% Training data**
* **20% Testing data**

A fixed `random_state=42` is used for reproducibility, and stratification is applied to preserve the target-class distribution.

## 📏 Feature Standardisation

The numerical features are standardised using `StandardScaler`.

Importantly, the scaler is fitted only on the training dataset and then applied to the test dataset to avoid test-set data leakage.

## 📊 Model Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix
* Classification Report

The models are compared using an evaluation table containing all major classification metrics.

## 🏆 Best Model Selection

The notebook selects the final model based on **F1 Score**.

```python
selection_metric = "F1 Score"
```

This selection criterion can be changed depending on the business objective. For example, recall could be prioritized when the objective is to identify as many potential converters as possible.

## 🔍 Model Interpretation

After selecting the final model, the project analyzes which features contribute most strongly to its predictions.

For tree-based models, feature importance is extracted using `feature_importances_`.

For Logistic Regression, the absolute value of the model coefficients is used as an importance proxy.

A confusion matrix and classification report are also generated for the selected model.

## 💰 Conversion Probability & Business Segmentation

The final model generates a conversion probability for test users.

Users are then segmented into three groups:

| Conversion Probability | Segment          |
| ---------------------- | ---------------- |
| 0–33%                  | Low Potential    |
| 33–66%                 | Medium Potential |
| 66–100%                | High Potential   |

This allows the machine learning output to be translated into a simple business-oriented segmentation.

## 📁 Repository Structure

```text
subscription-conversion-prediction-ml/
│
├── Subscription_Conversion_Prediction_ML_Project_Final.ipynb
├── subscription_conversion_100_users_dirty.csv
└── README.md
```

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Google Colab / Jupyter Notebook

## 📚 Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
```

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/subscription-conversion-prediction-ml.git
```

### 2. Open the notebook

Open:

```text
Subscription_Conversion_Prediction_ML_Project_Final.ipynb
```

using Jupyter Notebook, JupyterLab, or Google Colab.

### 3. Upload the dataset

Upload:

```text
subscription_conversion_100_users_dirty.csv
```

when prompted by the notebook.

### 4. Run all cells

The notebook will perform the complete workflow:

```text
Data Cleaning
→ EDA
→ Feature Engineering
→ Model Training
→ Evaluation
→ Best Model Selection
→ Prediction
→ Business Segmentation
```

## 📌 Key Output

The completed project produces:

* Cleaned dataset
* Missing-value treatment
* Outlier treatment
* EDA visualisations
* Engineered engagement features
* Trained classification models
* Model comparison metrics
* Confusion matrices
* Selected model based on F1 Score
* Feature importance
* Conversion probabilities
* Low/Medium/High potential customer segments

## 👨‍💻 Project

**Subscription Conversion Prediction — End-to-End Machine Learning Project**

Built to demonstrate the complete machine learning lifecycle from raw user data to actionable subscription-conversion predictions.
