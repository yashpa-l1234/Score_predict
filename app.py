import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Student Performance & Score Predictor",
    page_icon="🎓",
    layout="wide",
)

# Custom CSS for Shadow Effects and Modern Styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .card {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 6px 15px rgba(99, 102, 241, 0.3);
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Load the model securely with caching
@st.cache_resource
def load_model():
  try:
    with open("knn.pkl", "rb") as f:
      model = pickle.load(f)
    return model
  except Exception as e:
    st.error(f"Error loading model file: {e}")
    return None


model = load_model()

# Header Section
st.markdown(
    """
    <div class="card" style="text-align: center;">
        <h1>🎓 Student Academic Performance Predictor</h1>
        <p style="color: #6b7280; font-size: 1.1rem;">
            Provide the student parameters below to estimate performance using your trained model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if model is None:
  st.warning(
      "Please ensure 'knn.pkl' is uploaded in the same directory as `app.py`."
  )
else:
  with st.form("prediction_form"):
    st.markdown("### 📋 Enter Student Details")

    col1, col2, col3 = st.columns(3)

    with col1:
      age = st.number_input("Age", min_value=15, max_value=50, value=20)
      gender = st.selectbox("Gender", ["Male", "Female", "Other"])
      course = st.selectbox(
          "Course / Major",
          [
              "Computer Science",
              "Engineering",
              "Business",
              "Arts & Humanities",
              "Science",
          ],
      )
      study_hours = st.number_input(
          "Weekly Study Hours", min_value=0.0, max_value=80.0, value=15.0
      )

    with col2:
      class_attendance = st.slider(
          "Class Attendance (%)", min_value=0.0, max_value=100.0, value=85.0
      )
      internet_access = st.selectbox(
          "Internet Access at Home", ["Yes", "No"]
      )
      sleep_hours = st.number_input(
          "Average Sleep Hours (per night)",
          min_value=1.0,
          max_value=15.0,
          value=7.0,
      )
      sleep_quality = st.selectbox(
          "Sleep Quality", ["Poor", "Fair", "Good", "Excellent"]
      )

    with col3:
      study_method = st.selectbox(
          "Primary Study Method",
          [
              "Self-study",
              "Group Study",
              "Online Videos",
              "Flashcards / Practice",
          ],
      )
      facility_rating = st.slider(
          "Campus Facility Rating (1-10)", min_value=1, max_value=10, value=7
      )
      exam_difficulty = st.selectbox(
          "Expected Exam Difficulty", ["Easy", "Medium", "Hard"]
      )

    st.markdown("<br>", unsafe_allow_html=True)
    submit_btn = st.form_submit_button(
        "🚀 Predict Performance", use_container_width=True
    )

  if submit_btn:
    # --- ENCODE CATEGORICAL INPUTS TO MATCH NUMERIC MODEL EXPECTATIONS ---
    # Adjust these mappings if your training notebook used different numerical encoding (e.g., LabelEncoder indices).

    gender_val = 1 if gender == "Male" else (0 if gender == "Female" else 2)

    course_mapping = {
        "Computer Science": 0,
        "Engineering": 1,
        "Business": 2,
        "Arts & Humanities": 3,
        "Science": 4,
    }
    course_val = course_mapping.get(course, 0)

    internet_val = 1 if internet_access == "Yes" else 0

    sleep_quality_mapping = {
        "Poor": 0,
        "Fair": 1,
        "Good": 2,
        "Excellent": 3,
    }
    sleep_quality_val = sleep_quality_mapping.get(sleep_quality, 1)

    study_method_mapping = {
        "Self-study": 0,
        "Group Study": 1,
        "Online Videos": 2,
        "Flashcards / Practice": 3,
    }
    study_method_val = study_method_mapping.get(study_method, 0)

    exam_difficulty_mapping = {"Easy": 0, "Medium": 1, "Hard": 2}
    exam_difficulty_val = exam_difficulty_mapping.get(exam_difficulty, 1)

    # Build the numeric feature array (total 11 features)
    input_data = np.array([[
        age,
        gender_val,
        course_val,
        study_hours,
        class_attendance,
        internet_val,
        sleep_hours,
        sleep_quality_val,
        study_method_val,
        facility_rating,
        exam_difficulty_val,
    ]])

    try:
      prediction = model.predict(input_data)
      st.markdown("<br>", unsafe_allow_html=True)
      st.markdown(
          f"""
                <div class="metric-card">
                    <h2>Estimated Predicted Score</h2>
                    <p style="font-size: 2.5rem; font-weight: bold; margin: 0;">{prediction[0]:.2f}</p>
                </div>
                """,
          unsafe_allow_html=True,
      )
    except Exception as e:
      st.error(
          f"Prediction Error: {e}. \n\nTip: If your training notebook used a"
          " specific `LabelEncoder` object for text categories, save and load"
          " that encoder using pickle rather than manual dictionaries."
      )
