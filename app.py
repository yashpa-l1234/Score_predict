import streamlit as st
import joblib
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom Styling / Shadow and Layout enhancements
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #F3F4F6;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained KNN model
@st.cache_resource
def load_model():
    return joblib.load('knn.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Header Section
st.markdown('<div class="main-header">🎓 Student Performance Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Provide the student details below to estimate performance outcomes using a K-Nearest Neighbors (KNN) model.</div>', unsafe_allow_html=True)

# Input Form Layout
with st.form("prediction_form"):
    st.subheader("📝 Student Details & Habits")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=15, max_value=50, value=20, step=1)
        gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
        course = st.selectbox("Course / Field of Study", options=["Science", "Arts", "Engineering", "Business", "Other"])
        study_hours = st.slider("Daily Study Hours", min_value=0.0, max_value=15.0, value=4.0, step=0.5)
        class_attendance = st.slider("Class Attendance (%)", min_value=0, max_value=100, value=85, step=1)

    with col2:
        internet_access = st.selectbox("Internet Access at Home", options=["Yes", "No"])
        sleep_hours = st.slider("Daily Sleep Hours", min_value=0.0, max_value=12.0, value=7.0, step=0.5)
        sleep_quality = st.selectbox("Sleep Quality", options=["Good", "Average", "Poor"])
        study_method = st.selectbox("Primary Study Method", options=["Self-Study", "Group Study", "Online Videos", "Classes"])
        facility_rating = st.slider("Campus Facility Rating (1-10)", min_value=1, max_value=10, value=7, step=1)

    submitted = st.form_submit_button("Predict Performance", use_container_width=True)

# Prediction Logic
if submitted:
    # Note: Depending on how your categorical columns were encoded during training 
    # (e.g., One-Hot Encoding vs Label Encoding), you may need to transform 
    # these raw user inputs into the exact numerical array format the model expects.
    
    # Placeholder mapping example (Adjust indices based on your training pipeline):
    gender_val = 1 if gender == "Male" else (0 if gender == "Female" else 2)
    internet_val = 1 if internet_access == "Yes" else 0
    
    # Construct feature array matching the 10 training features[cite: 1]
    # (Modify this feature vector assembly to match your exact preprocessing pipeline)
    features = np.array([[
        age, 
        gender_val, 
        0, # placeholder categorical encoding for course
        study_hours, 
        class_attendance, 
        internet_val, 
        sleep_hours, 
        0, # placeholder categorical encoding for sleep_quality
        0, # placeholder categorical encoding for study_method
        facility_rating
    ]])

    try:
        prediction = model.predict(features)
        
        # Display Result Card with Shadow Effect
        st.markdown(f"""
            <div class="prediction-card">
                <h3>Predicted Result</h3>
                <h1 style="color: #2563EB;">{prediction[0]:.2f}</h1>
            </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Prediction failed. Ensure your feature array matches the model's expected shape. Error: {e}")
