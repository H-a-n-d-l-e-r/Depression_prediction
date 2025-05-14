import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os

def prepare_data(df, scaler):
    """Prepare the data for prediction using the pre-trained scaler"""
    # Ensure we have all the required features in the correct order
    required_features = [
        'Age', 'Academic_Year', 'Current_CGPA', 'waiver_or_scholarship',
        'PSS1', 'PSS2', 'PSS3', 'PSS4', 'PSS5', 'PSS6', 'PSS7', 'PSS8', 'PSS9', 'PSS10',
        'GAD1', 'GAD2', 'GAD3', 'GAD4', 'GAD5', 'GAD6', 'GAD7',
        'Gender_Female', 'Gender_Male', 'Gender_Prefer_not_to_say',
        'Department_Biological_Sciences',
        'Department_Business_and_Entrepreneurship_Studies',
        'Department_Engineering_-_CS_/_CSE_/_CSC_/_Similar_to_CS',
        'Department_Engineering_-_Civil_Engineering_/_Similar_to_CE',
        'Department_Engineering_-_EEE/_ECE_/_Similar_to_EEE',
        'Department_Engineering_-_Mechanical_Engineering_/_Similar_to_ME',
        'Department_Engineering_-_Other',
        'Department_Environmental_and_Life_Sciences',
        'Department_Law_and_Human_Rights',
        'Department_Liberal_Arts_and_Social_Sciences',
        'Department_Other',
        'Department_Pharmacy_and_Public_Health'
    ]
    
    # Ensure all required features are present
    for feature in required_features:
        if feature not in df.columns:
            df[feature] = 0
    
    # Select only the required features in the correct order
    df = df[required_features]
    
    # Transform using the pre-trained scaler
    X_scaled = scaler.transform(df)
    
    return X_scaled

# Set page config
st.set_page_config(
    page_title="Depression Assessment Tool",
    page_icon="🧠",
    layout="wide"
)

# Title and description
st.title("Depression Assessment Tool")
st.markdown("""
This tool helps assess depression levels based on various factors. 
Please answer all questions honestly to get an accurate assessment.
""")

# Load the trained model and scaler
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open('xgboost_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
        return model, scaler, label_encoder
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

# Function to create the questionnaire
def create_questionnaire():
    st.header("Personal Information")
    
    # Basic information
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=16, max_value=100, value=20)
        academic_year = st.selectbox("Academic Year", [1, 2, 3, 4, 5, 6])
        current_cgpa = st.number_input("Current CGPA", min_value=0.0, max_value=4.0, value=3.0, step=0.01)
    
    with col2:
        waiver = st.selectbox("Do you have a waiver or scholarship?", ["No", "Yes"])
        gender = st.selectbox("Gender", ["Male", "Female", "Prefer not to say"])
        department = st.selectbox("Department", [
            "Biological Sciences",
            "Business and Entrepreneurship Studies",
            "Engineering - CS/CSE/CSC/Similar to CS",
            "Engineering - Civil Engineering/Similar to CE",
            "Engineering - EEE/ECE/Similar to EEE",
            "Engineering - Mechanical Engineering/Similar to ME",
            "Engineering - Other",
            "Environmental and Life Sciences",
            "Law and Human Rights",
            "Liberal Arts and Social Sciences",
            "Other",
            "Pharmacy and Public Health"
        ])

    # Convert categorical variables to binary
    gender_female = 1 if gender == "Female" else 0
    gender_male = 1 if gender == "Male" else 0
    gender_prefer_not_to_say = 1 if gender == "Prefer not to say" else 0
    waiver_or_scholarship = 1 if waiver == "Yes" else 0

    # Create department binary variables
    dept_columns = [
        "Department_Biological_Sciences",
        "Department_Business_and_Entrepreneurship_Studies",
        "Department_Engineering_-_CS_/_CSE_/_CSC_/_Similar_to_CS",
        "Department_Engineering_-_Civil_Engineering_/_Similar_to_CE",
        "Department_Engineering_-_EEE/_ECE_/_Similar_to_EEE",
        "Department_Engineering_-_Mechanical_Engineering_/_Similar_to_ME",
        "Department_Engineering_-_Other",
        "Department_Environmental_and_Life_Sciences",
        "Department_Law_and_Human_Rights",
        "Department_Liberal_Arts_and_Social_Sciences",
        "Department_Other",
        "Department_Pharmacy_and_Public_Health"
    ]
    
    dept_dict = {col: 0 for col in dept_columns}
    dept_dict[f"Department_{department.replace(' ', '_').replace('-', '_').replace('/', '_')}"] = 1

    # PSS Questions
    st.header("Stress Assessment (PSS)")
    st.markdown("""
    Please indicate how often you have felt or thought each of the following in the last month:
    (0 = Never, 1 = Almost Never, 2 = Sometimes, 3 = Fairly Often, 4 = Very Often)
    """)
    
    pss_questions = {
        "PSS1": "In the last month, how often have you been upset because of something that happened unexpectedly?",
        "PSS2": "In the last month, how often have you felt that you were unable to control the important things in your life?",
        "PSS3": "In the last month, how often have you felt nervous and stressed?",
        "PSS4": "In the last month, how often have you felt confident about your ability to handle your personal problems?",
        "PSS5": "In the last month, how often have you felt that things were going your way?",
        "PSS6": "In the last month, how often have you found that you could not cope with all the things that you had to do?",
        "PSS7": "In the last month, how often have you been able to control irritations in your life?",
        "PSS8": "In the last month, how often have you felt that you were on top of things?",
        "PSS9": "In the last month, how often have you been angered because of things that happened that were outside of your control?",
        "PSS10": "In the last month, how often have you felt difficulties were piling up so high that you could not overcome them?"
    }

    pss_answers = {}
    for q_id, question in pss_questions.items():
        pss_answers[q_id] = st.slider(question, 0, 4, 2)

    # GAD Questions
    st.header("Anxiety Assessment (GAD)")
    st.markdown("""
    Please indicate how often you have been bothered by each of the following problems in the last two weeks:
    (0 = Not at all, 1 = Several days, 2 = More than half the days, 3 = Nearly every day)
    """)
    
    gad_questions = {
        "GAD1": "Feeling nervous, anxious, or on edge",
        "GAD2": "Not being able to stop or control worrying",
        "GAD3": "Worrying too much about different things",
        "GAD4": "Trouble relaxing",
        "GAD5": "Being so restless that it's hard to sit still",
        "GAD6": "Becoming easily annoyed or irritable",
        "GAD7": "Feeling afraid as if something awful might happen"
    }

    gad_answers = {}
    for q_id, question in gad_questions.items():
        gad_answers[q_id] = st.slider(question, 0, 3, 1)

    # Create input data
    input_data = {
        'Age': age,
        'Academic_Year': academic_year,
        'Current_CGPA': current_cgpa,
        'waiver_or_scholarship': waiver_or_scholarship,
        **pss_answers,
        **gad_answers,
        'Gender_Female': gender_female,
        'Gender_Male': gender_male,
        'Gender_Prefer_not_to_say': gender_prefer_not_to_say,
        **dept_dict
    }

    return pd.DataFrame([input_data])

# Main app
def main():
    # Load model
    model, scaler, label_encoder = load_model()
    if model is None:
        return

    # Create questionnaire
    input_df = create_questionnaire()

    # Add predict button
    if st.button("Get Assessment"):
        with st.spinner("Analyzing your responses..."):
            try:
                # Scale the input data using the pre-trained scaler
                input_scaled = prepare_data(input_df, scaler)
                
                # Make prediction
                prediction = model.predict(input_scaled)
                prediction_proba = model.predict_proba(input_scaled)
                
                # Get the predicted class and its probability
                predicted_class = label_encoder.inverse_transform(prediction)[0]
                confidence = prediction_proba[0][prediction[0]] * 100
                
                # Display results
                st.header("Assessment Results")
                st.markdown(f"""
                ### Predicted Depression Level: {predicted_class}
                Confidence: {confidence:.1f}%
                """)
                
                # Add interpretation
                st.markdown("""
                ### Interpretation
                - **No Depression**: No significant symptoms
                - **Minimal Depression**: Very mild symptoms
                - **Mild Depression**: Some symptoms present
                - **Moderate Depression**: Several symptoms present
                - **Moderately Severe Depression**: Many symptoms present
                - **Severe Depression**: Most symptoms present
                
                Please note that this is not a medical diagnosis. If you're concerned about your mental health,
                please consult with a healthcare professional.
                """)
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.write("Input DataFrame shape:", input_df.shape)
                st.write("Input DataFrame columns:", input_df.columns.tolist())

if __name__ == "__main__":
    main() 