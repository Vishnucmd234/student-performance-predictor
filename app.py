import streamlit as st
import joblib
import numpy as np
import warnings

# 1. वर्जन वाली वार्निंग छुपाने के लिए (Optional)
warnings.filterwarnings("ignore", category=UserWarning)

# 1. Model aur Scaler load karein
model = joblib.load('st_model.pkl')
scaler = joblib.load('st_scaler.pkl')

st.title("🎓 Student Performance Predictor")
st.write("Apne details bhariye niche prediction dekhne ke liye:")

# 2. User Input Lena
col1, col2 = st.columns(2)

with col1:
    hours_studied = st.number_input("Hours Studied (0-9)", min_value=0, max_value=9, value=5)
    prev_scores = st.number_input("Previous Scores (0-100)", min_value=0, max_value=100, value=70)
    sleep_hours = st.number_input("Sleep Hours (1-9)", min_value=1, max_value=9, value=7)

with col2:
    # Extracurricular Activities ko 1 (Yes) aur 0 (No) mein badalna
    extra_act = st.selectbox("Extracurricular Activities", ["Yes", "No"])
    extra_act_val = 1 if extra_act == "Yes" else 0
    
    sample_papers = st.number_input("Sample Question Papers Practiced (1-9)", min_value=1, max_value=9, value=3)

# 3. Prediction Button
if st.button("Predict Performance Score"):
    # Input data ko array mein convert karna (Wahi order jo model ne seekha hai) 
    input_data = np.array([[hours_studied, prev_scores, extra_act_val, sleep_hours, sample_papers]])
    
    # Data ko scale karna (Fit nahi, sirf transform!)
    scaled_data = scaler.transform(input_data)
    
    # Model se prediction lena [cite: 4]
    prediction = model.predict(scaled_data)
    
    # Result dikhana
    st.subheader(f"Expected Performance Score: {prediction[0]:.2f}")
    
    # Thodi extra details
    if prediction[0] > 80:
        st.balloons()
        st.success("Bohot badhiya score! Aise hi mehnat karte rahiye.")
    elif prediction[0] > 50:
        st.info("Achha score hai, thodi aur mehnat se ye aur behtar ho sakta hai.")
    else:
        st.warning("Aapko thoda aur dhyan dene ki zaroorat hai.")