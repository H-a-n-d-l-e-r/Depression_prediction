# ML_Project_Student_Mental_Health_Prediction

📂 [Access the Dataset on Google Drive]((https://drive.google.com/drive/folders/1z_VZuXSdXW84y_0Bwc-s-PWu4YlfmKjI?usp=share_link))


## 🧠 Project Overview

This project aims to explore and predict levels of depression among university students using survey data collected through a detailed questionnaire. Through data cleaning, feature engineering, statistical analysis, and machine learning, the goal is to identify key factors contributing to depression and assess the feasibility of automating mental health risk classification.

## 📊 Dataset

Source

The data was collected via a custom-built questionnaire answered by university students in Bangladesh. The full questionnaire is available in Supplementary file - Questionnaire.pdf.

Structure

Total responses: 2022 students

Columns: PHQ-9 (Depression), GAD-7 (Anxiety), PSS-10 (Perceived Stress), and academic/demographic variables

Labels

Depression Label: Derived from the total PHQ-9 score using standard thresholds

Additional derived scores: Anxiety_Value, Stress_Value, and corresponding labels (not used in modeling to avoid leakage)

## 🧹 Data Preparation

Detailed in the notebook Stress_data_cleaning.ipynb:

Converted categorical ranges (e.g., Age, CGPA) to numerical values

One-hot encoded Gender and Department columns

Handled missing values (e.g., imputed GPA based on depression scores)

Removed PHQ items to prevent data leakage when predicting Depression_Label

Final cleaned dataset saved as Cleaned_Final.csv

## 📈 Exploratory Data Analysis

See Stress_data_analysis.ipynb:

Correlation heatmaps for PHQ, GAD, and PSS scores

Bar plots showing average depression, anxiety, and stress by gender, CGPA, department, and academic year

KDE plots showing score distributions by gender

## 🤖 Machine Learning Methods

Notebook: Stress_ML_1.ipynb

We compared the performance of three models:

Logistic Regression

Random Forest Classifier

XGBoost Classifier

Target variable: Depression_Label (6-class classification)

Key steps:

Removed label-derived columns to prevent data leakage

Used only GAD/PSS + academic/demographic features

Train-test split (80/20), stratified by class

Scaled numeric features with StandardScaler

Evaluated models using accuracy, weighted F1, and classification report

Visualized top feature importances for Random Forest and XGBoost

## 📦 Dependencies

This project uses the following Python libraries:

pandas, numpy

matplotlib, seaborn

scikit-learn

xgboost

(optional for profiling) ydata-profiling

## 🔁 Reproducibility

To reproduce the analysis and results:

Clone this repository

Install the required packages using:

pip install -r requirements.txt

Open and run the notebooks in order:

Stress_data_cleaning.ipynb

Stress_data_analysis.ipynb

Stress_ML_1.ipynb

Ensure Cleaned_Final.csv is present in your working directory for the modeling notebook.

## 🙋‍♂️ Author

This project was developed as part of a machine learning course by a university student interested in mental health analytics and applied AI. Reach out via GitHub for questions or collaborations.

# Depression Assessment Tool

This is a Streamlit web application that helps assess depression levels based on various factors including stress and anxiety levels.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Save the model and necessary files:
```bash
python save_model.py
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

## Features

- Interactive questionnaire for personal information
- Stress assessment (PSS)
- Anxiety assessment (GAD)
- Real-time depression level prediction
- Confidence score for the prediction
- Interpretation of results

## Important Note

This tool is not a medical diagnosis. If you're concerned about your mental health, please consult with a healthcare professional.

## File Structure

- `app.py`: The main Streamlit application
- `save_model.py`: Script to train and save the model
- `requirements.txt`: Required Python packages
- `xgboost_model.pkl`: Saved XGBoost model
- `scaler.pkl`: Saved StandardScaler
- `label_encoder.pkl`: Saved LabelEncoder
- `Cleaned_Final.csv`: Dataset used for training

## Usage

1. Fill out the personal information section
2. Complete the stress assessment (PSS) questions
3. Complete the anxiety assessment (GAD) questions
4. Click "Get Assessment" to see your results
5. Review the interpretation of your results

## Model Information

The model uses XGBoost classifier with the following features:
- Personal information (age, academic year, CGPA, etc.)
- Stress assessment scores (PSS)
- Anxiety assessment scores (GAD)
- Department and gender information

The model predicts one of six depression levels:
- No Depression
- Minimal Depression
- Mild Depression
- Moderate Depression
- Moderately Severe Depression
- Severe Depression

