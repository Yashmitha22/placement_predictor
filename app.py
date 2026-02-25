import streamlit as st
import pickle
import numpy as np

# Page Configuration
st.set_page_config(page_title="Placement Predictor", page_icon="🎓")

# Title
st.title("🎓 Placement Prediction System")
st.markdown("### Enter Student Details to Predict Placement")

# Load the ML Model
@st.cache_resource
def load_model():
    with open('placement_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
    st.success("✅ Model loaded successfully!")
except FileNotFoundError:
    st.error("❌ Model file not found! Please ensure 'placement_model.pkl' exists.")
    st.stop()

# Sidebar for additional info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This app predicts placement status based on IQ and CGPA.")
    st.write("Model: Random Forest Classifier")
    st.write("Accuracy: 85%")  # Update with your model's accuracy

# Input Fields
st.subheader("📝 Student Information")

col1, col2 = st.columns(2)

with col1:
    iq = st.number_input(
        "IQ Score", 
        min_value=50, 
        max_value=200, 
        value=100,
        help="Enter IQ score between 50-200"
    )

with col2:
    cgpa = st.number_input(
        "CGPA (out of 10)", 
        min_value=0.0, 
        max_value=10.0, 
        value=7.0, 
        step=0.01,
        help="Enter CGPA between 0.0-10.0"
    )

# Optional: Add more features if your model uses them
# with col3:
#     attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=75)

# Prediction Button
if st.button("🔮 Predict Placement", type="primary"):
    # Prepare input data
    input_data = np.array([[iq, cgpa]])  # Adjust shape based on your model
    
    # Make prediction
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0]
    
    # Display Results
    st.subheader("📊 Prediction Result")
    
    if prediction[0] == 1:
        st.success(f"**Result: PLACED ✅**")
        st.metric("Confidence", f"{probability[1]*100:.2f}%")
    else:
        st.error(f"**Result: NOT PLACED ❌**")
        st.metric("Confidence", f"{probability[0]*100:.2f}%")
    
    # Show probability breakdown
    st.markdown("### Probability Distribution")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"Placed: **{probability[1]*100:.2f}%**")
    with col2:
        st.warning(f"Not Placed: **{probability[0]*100:.2f}%**")
    
    # Input Summary
    with st.expander("📋 View Input Summary"):
        st.write(f"**IQ Score:** {iq}")
        st.write(f"**CGPA:** {cgpa}")

