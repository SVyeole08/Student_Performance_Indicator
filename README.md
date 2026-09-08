# Student Performance Indicator 🎓

Student Performance Indicator is an end-to-end machine learning web application that predicts a student's **Math Score**
from demographic, academic, and test preparation attributes. The project covers data ingestion, preprocessing, model
evaluation, hyperparameter tuning, and production inference through a Flask backend.

🔗 [**Live Application**](https://student-performance-indicator-3a1l.onrender.com/)

---

## Architecture & Pipeline

```text
Student Input (HTML Form)
         ↓
Flask Route Handling & Data Preparation
         ↓
Feature Transformation (Preprocessor artifact)
         ↓
Best Regression Model Inference
         ↓
Rendered Math Score Prediction
```

---

## Features & Model Details

The model predicts **Math Score** based on 7 input features:

| Feature                       | Type        | Description                    |
|:------------------------------|:------------|:-------------------------------|
| `gender`                      | Categorical | Student's gender               |
| `race_ethnicity`              | Categorical | Student's race/ethnicity group |
| `parental_level_of_education` | Categorical | Parent's education level       |
| `lunch`                       | Categorical | Type of lunch                  |
| `test_preparation_course`     | Categorical | Test preparation course status |
| `reading_score`               | Integer     | Reading test score             |
| `writing_score`               | Integer     | Writing test score             |

### Performance

* **Target:** Math Score
* **Evaluation Metric:** R² Score : 0.8805931485028737 (Ridge Regression)
* **Models Evaluated:** Linear Regression, Ridge, Lasso, KNN, SVR, Decision Tree, Random Forest, AdaBoost, Gradient
  Boosting, XGBoost, CatBoost

---

## 📁 Project Structure

```text
Student_Performance_Indicator/
├── app.py                         # Flask backend & prediction routes
├── artifact/                      # Artifacts
│   ├── data.csv
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── test.csv
│   └── train.csv
├── src/
│   ├── components/
│   │   ├── data_ingestion.py      # Data ingestion & train/test split
│   │   ├── data_transformation.py # Feature preprocessing
│   │   └── model_trainer.py       # Model training & evaluation
│   ├── pipeline/
│   │   └── predict_pipeline.py    # Prediction pipeline
│   ├── exception.py               # Custom exception handling
│   ├── logger.py                  # Application logging
│   └── utils.py                   # Utility functions
├── templates/
│   ├── home.html                  # Prediction form
│   └── index.html                 # Landing page
├── requirements.txt               # Python dependencies
├── setup.py                       # Package configuration
├── Dockerfile                     # Container configuration
└── README.md
```

---

## Quickstart (Run Locally)

### 1. Clone the repository

```bash
git clone https://github.com/SVyeole08/Student_Performance_Indicator.git
cd Student_Performance_Indicator
```

### 2. Set up environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Start the application

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

---

## Tech Stack

* **Core Language:** Python
* **Data Science & ML:** Scikit-learn, Pandas, NumPy, XGBoost, CatBoost, Matplotlib, Seaborn
* **Web Framework:** Flask, Jinja2
* **Model Persistence:** Dill
* **WSGI & Deployment:** Gunicorn, Docker

---

##  Model Training

The training pipeline can be executed with:

```bash
python -m src.components.data_ingestion
```

This runs data ingestion, transformation, model evaluation, hyperparameter tuning, and best model selection.

---

<div style="font-size: 15px" align="center">
⭐ If this was useful, a star helps other people find it.
</div>
