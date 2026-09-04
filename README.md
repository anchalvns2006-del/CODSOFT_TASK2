# 🛡️ FraudGuard AI

### AI-Powered Credit Card Fraud Detection System

FraudGuard AI is an intelligent machine learning application designed to detect potentially fraudulent credit card transactions.

The system uses a **Random Forest Classifier** trained on historical transaction data and provides an interactive **Streamlit dashboard** for analytics, fraud prediction, and model evaluation.

---

## 🚀 Live Demo

🔗 **[Try FraudGuard AI Live](https://anchalvns2006-del-codsoft-task2-app-0k8yey.streamlit.app/)**

---

## ✨ Features

- 🏠 **Interactive Dashboard**
- 📊 **Fraud Analytics**
- 🔮 **Real-Time Fraud Prediction**
- 🤖 **Model Performance Analysis**
- 🌲 Random Forest Classification
- 📈 ROC-AUC Analysis
- 🔢 Confusion Matrix
- 🎯 Feature Importance
- 💰 Transaction Amount Analysis
- 🕐 Time-Based Fraud Analysis
- 📍 Transaction Distance Analysis
- 🛒 Category-Based Fraud Analysis
- 🌙 Night vs Day Fraud Analysis
- 🗓️ Weekend vs Weekday Analysis
- 🎨 Professional Streamlit UI

---

## 📊 Dataset

The project uses a credit card transaction dataset containing:

| Information | Value |
|---|---:|
| Total Transactions | 555,719 |
| Legitimate Transactions | 553,574 |
| Fraud Transactions | 2,145 |
| Fraud Rate | 0.39% |
| Features Used | 21+ |

The dataset contains transaction, customer, merchant, geographical and time-related information.

---

## 🧠 Machine Learning

### Algorithm

**Random Forest Classifier**

The model was trained after performing:

- Data cleaning
- Feature engineering
- Categorical encoding
- Train/Test splitting
- Feature transformation
- Model training
- Model evaluation

### Engineered Features

Some important engineered features include:

- Transaction Hour
- Customer Age
- Transaction Distance
- Day of Week
- Month
- Weekend Indicator
- Night Transaction Indicator
- Card Transaction Count
- Average Card Amount
- Amount Deviation
- Large Distance Indicator
- High Amount Indicator

---

## 📈 Model Performance

| Metric | Score |
|---|---:|
| Accuracy | **99.83%** |
| Precision | **95%** |
| Recall | **58%** |
| F1-Score | **72%** |
| ROC-AUC | **96.95%** |

> Because the dataset is highly imbalanced, accuracy alone is not sufficient. Precision, recall, F1-score and ROC-AUC are also considered for evaluation.

---

## 📊 Analytics Dashboard

FraudGuard AI provides interactive visualizations for understanding fraud behaviour.

### Available Analysis

**Transaction Analysis**
- Transaction distribution
- Transaction amount distribution
- Fraud by amount range

**Time Analysis**
- Fraud by hour
- Fraud by day
- Fraud by month
- Weekend vs weekday
- Night vs day

**Customer & Merchant Analysis**
- Fraud by gender
- Fraud by category
- Transaction distance

**Statistical Analysis**
- Feature correlation
- Feature importance

---

## 🔮 Fraud Prediction

The prediction module allows users to enter transaction details and receive an AI-based risk assessment.

### Input Parameters

- 💰 Transaction Amount
- 🕐 Transaction Hour
- 👤 Customer Age
- 📍 Merchant Distance
- 🏙️ City Population
- 🗓️ Weekend Status

The application returns:

- Fraud / Legitimate prediction
- Fraud probability
- Risk status

---

## 🖥️ Application Pages

### 🏠 Dashboard

Provides a high-level overview of transaction activity and fraud statistics.

### 📊 Analytics

Provides interactive charts and detailed fraud behaviour analysis.

### 🔮 Fraud Prediction

Allows users to analyze an individual transaction.

### 🤖 Model Performance

Displays:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Feature Importance

---

## 🛠️ Tech Stack

### Programming

- Python

### Machine Learning

- Scikit-learn
- Random Forest

### Data Processing

- Pandas
- NumPy

### Visualization

- Plotly

### Application

- Streamlit

### Model Serialization

- Joblib

### Version Control

- Git
- GitHub

---

## 📂 Project Structure

```text
CODSOFT_TASK2/
│
├── app.py
├── style.css
├── requirements.txt
├── README.md
│
└── notebooks/
    ├── fraud_detection_model.pkl
    └── fraudTest.csv