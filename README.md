# ❤️ Heart Stroke Risk Prediction

This project is a **machine learning-powered web app** that predicts the likelihood of heart disease based on user-provided health parameters.  
It uses a **Logistic Regression** model trained on cardiovascular data, selected after evaluating multiple supervised classification algorithms.

---

## 📋 Features

- User-friendly **Streamlit** web interface
- Accepts health parameters like age, cholesterol, blood pressure, heart rate, etc.
- Uses the same preprocessing pipeline from training (scaler + expected columns)
- Real-time prediction with a clear **High Risk / Low Risk** output message

---

## 📊 Model Selection

We tested the following supervised classification algorithms:

1. Logistic Regression ✅ _(best performer)_
2. K-Nearest Neighbors (KNN)
3. Support Vector Machine (SVM)
4. Decision Tree
5. Naive Bayes

Logistic Regression was chosen based on the best performance in terms of **accuracy** and **f1_score**.

---

## 📦 Installation & Running Locally

### 1️⃣ Clone this repository

```bash
git clone https://github.com/Ariz253/HeartWise.git
cd HeartWise
```

### 2️⃣ Create a virtual environment (recommended)

```bash
python -m venv venv
# On Mac/Linux
source venv/bin/activate
# On Windows
venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Ensure model files are present

The following files should be in the project root:

- `heart_disease_model.pkl` → trained Logistic Regression model
- `heart_disease_scaler.pkl` → scaler object used in preprocessing
- `heart_disease_columns.pkl` → expected column names for input data

### 5️⃣ Run the Streamlit app

```bash
streamlit run app.py
```
