import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import json
import os
import re
from PIL import Image
import numpy as np

# Set pandas options to handle large DataFrames
pd.set_option("styler.render.max_elements", 6716325)

# Set page configuration
st.set_page_config(
    page_title="🏥 Diabetic Readmission Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize users.json if it doesn't exist
if not os.path.exists('users.json'):
    with open('users.json', 'w') as f:
        json.dump([], f)

# User authentication functions
def load_users():
    try:
        with open('users.json', 'r') as f:
            users = json.load(f)
            return [user for user in users if all(key in user for key in ['username', 'password', 'email', 'hospital', 'role'])]
    except:
        return []

def save_users(users):
    with open('users.json', 'w') as f:
        json.dump(users, f)

def is_valid_email(email):
    """Validate the basic format of an email address."""
    if not isinstance(email, str):
        return False

    email = email.strip()

    pattern = r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$"

    return re.fullmatch(pattern, email) is not None

def register_user(username, password, email, hospital, role):
    users = load_users()
    email = email.strip().lower()
    username = username.strip()
    
    if not all([username, password, email, hospital, role]):
        return False, "All fields are required"
    if not is_valid_email(email):
        return False, "Invalid email format. Please enter a valid email address (e.g. name@example.com)."
    
    for user in users:
        if user['username'].strip().lower() == username.lower():
            return False, "Username already exists"
        if user['email'].strip().lower() == email:
            return False, "Email already registered"
    
    users.append({
        'username': username,
        'password': password,
        'email': email,
        'hospital': hospital,
        'role': role
    })
    
    save_users(users)
    return True, "Registration successful"
    

def login_user(username, password):
    users = load_users()
    for user in users:
        if user['username'] == username and user['password'] == password:
            return True, user
    return False, None

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_info' not in st.session_state:
    st.session_state.user_info = None
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

# Outcome styling
OUTCOME_COLORS = {
    "No": "#2ecc71",  # Green
    "Yes": "#e74c3c"   # Red
}

OUTCOME_EMOJIS = {
    "No": "🟢",
    "Yes": "🔴"
}

@st.cache_resource
def load_model():
    return joblib.load("readmission_ML.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("Karanda-Diabetic-Patients-Processed.csv")
    # Optimize memory usage by downcasting numeric columns
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    return df

# Initialize model and data
model = load_model()
data = load_data()

def get_input_features(data):
    """Get input features with comprehensive validation"""
    with st.form("prediction_form"):
        st.header("Patient Readmission Risk Assessment")
        st.markdown("Please provide patient information to assess readmission risk:")
        st.markdown("**⚠️ All fields are required**")
        
        # Create two main columns
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.subheader("Demographic Information")
            race = st.selectbox('Race *', data['Race'].unique(), index=None, placeholder="Select Race...")
            gender = st.selectbox('Gender *', data['Gender'].unique(), index=None, placeholder="Select Gender...")
            age = st.selectbox('Age *', data['Age'].unique(), index=None, placeholder="Select Age...")
            time_in_hospital = st.number_input('Time in Hospital (days) *', 
                                            min_value=0, max_value=100, value=None, step=1,
                                            placeholder="Enter number of days...")
            speciality = st.selectbox('Speciality *', data['Speciality'].unique(), index=None, placeholder="Select Speciality...")
            
        with col2:
            st.subheader("Medical Information")
            subcol1, subcol2 = st.columns(2)
            
            with subcol1:
                max_glu_serum = st.selectbox('Max Glucose Serum *', data['max_glu_serum'].unique(), index=None, placeholder="Select...")
                a1cresult = st.selectbox('A1C Result *', data['A1Cresult'].unique(), index=None, placeholder="Select...")
                diabetesmed = st.radio('Diabetes Medication *', ['Yes', 'No'], horizontal=True, index=None)
                
            with subcol2:
                insulin = st.selectbox('Insulin *', data['Insulin'].unique(), index=None, placeholder="Select...")
                metformin = st.selectbox('Metformin *', data['Metformin'].unique(), index=None, placeholder="Select...")
                
        st.subheader("Medications")
        st.markdown("**All medication fields are required**")
        med_col1, med_col2, med_col3, med_col4 = st.columns(4)
        
        with med_col1:
            repaglinide = st.selectbox('Repaglinide *', data['Repaglinide'].unique(), index=None, placeholder="Select...")
            nateglinide = st.selectbox('Nateglinide *', data['Nateglinide'].unique(), index=None, placeholder="Select...")
            chlorpropamide = st.selectbox('Chlorpropamide *', data['Chlorpropamide'].unique(), index=None, placeholder="Select...")
            glimepiride = st.selectbox('Glimepiride *', data['Glimepiride'].unique(), index=None, placeholder="Select...")
            acetohexamide = st.selectbox('Acetohexamide *', data['Acetohexamide'].unique(), index=None, placeholder="Select...")
            
        with med_col2:
            glipizide = st.selectbox('Glipizide *', data['Glipizide'].unique(), index=None, placeholder="Select...")
            glyburide = st.selectbox('Glyburide *', data['Glyburide'].unique(), index=None, placeholder="Select...")
            tolbutamide = st.selectbox('Tolbutamide *', data['Tolbutamide'].unique(), index=None, placeholder="Select...")
            pioglitazone = st.selectbox('Pioglitazone *', data['Pioglitazone'].unique(), index=None, placeholder="Select...")
            rosiglitazone = st.selectbox('Rosiglitazone *', data['Rosiglitazone'].unique(), index=None, placeholder="Select...")
            
        with med_col3:
            acarbose = st.selectbox('Acarbose *', data['Acarbose'].unique(), index=None, placeholder="Select...")
            miglitol = st.selectbox('Miglitol *', data['Miglitol'].unique(), index=None, placeholder="Select...")
            tolazamide = st.selectbox('Tolazamide *', data['Tolazamide'].unique(), index=None, placeholder="Select...")
            examide = st.selectbox('Examide *', data['Examide'].unique(), index=None, placeholder="Select...")
            citoglipton = st.selectbox('Citoglipton *', data['Citoglipton'].unique(), index=None, placeholder="Select...")
            
        with med_col4:
            troglitazone = st.selectbox('Troglitazone *', data['Troglitazone'].unique(), index=None, placeholder="Select...")
            glyburide_metformin = st.selectbox('Glyburide-metformin *', data['Glyburide-metformin'].unique(), index=None, placeholder="Select...")
            glipizide_metformin = st.selectbox('glipizide-metformin *', data['glipizide-metformin'].unique(), index=None, placeholder="Select...")
            glimepiride_pioglitazone = st.selectbox('Glimepiride-pioglitazone *', data['Glimepiride-pioglitazone'].unique(), index=None, placeholder="Select...")
            metformin_rosiglitazone = st.selectbox('metformin-rosiglitazone *', data['metformin-rosiglitazone'].unique(), index=None, placeholder="Select...")
            metformin_pioglitazone = st.selectbox('Metformin-pioglitazone *', data['Metformin-pioglitazone'].unique(), index=None, placeholder="Select...")
        
        st.markdown("---")
        st.markdown("**⚠️ Please ensure all fields marked with * are completed before submitting**")
        submit_button = st.form_submit_button("Predict Readmission Risk", 
                                            type="primary", 
                                            use_container_width=True)
        
        if submit_button:
            # Comprehensive validation - collect all missing fields
            missing_fields = []
            validation_errors = []
            
            # Check Demographic Information
            if race is None or race == "":
                missing_fields.append("Race")
            if gender is None or gender == "":
                missing_fields.append("Gender")
            if age is None or age == "":
                missing_fields.append("Age")
            if time_in_hospital is None or time_in_hospital == "":
                missing_fields.append("Time in Hospital")
            if speciality is None or speciality == "":
                missing_fields.append("Speciality")
            
            # Check Medical Information
            if max_glu_serum is None or max_glu_serum == "":
                missing_fields.append("Max Glucose Serum")
            if a1cresult is None or a1cresult == "":
                missing_fields.append("A1C Result")
            if diabetesmed is None or diabetesmed == "":
                missing_fields.append("Diabetes Medication")
            if insulin is None or insulin == "":
                missing_fields.append("Insulin")
            if metformin is None or metformin == "":
                missing_fields.append("Metformin")
            
            # Check all medication fields
            medication_fields = {
                'Repaglinide': repaglinide,
                'Nateglinide': nateglinide,
                'Chlorpropamide': chlorpropamide,
                'Glimepiride': glimepiride,
                'Acetohexamide': acetohexamide,
                'Glipizide': glipizide,
                'Glyburide': glyburide,
                'Tolbutamide': tolbutamide,
                'Pioglitazone': pioglitazone,
                'Rosiglitazone': rosiglitazone,
                'Acarbose': acarbose,
                'Miglitol': miglitol,
                'Tolazamide': tolazamide,
                'Examide': examide,
                'Citoglipton': citoglipton,
                'Troglitazone': troglitazone,
                'Glyburide-metformin': glyburide_metformin,
                'glipizide-metformin': glipizide_metformin,
                'Glimepiride-pioglitazone': glimepiride_pioglitazone,
                'metformin-rosiglitazone': metformin_rosiglitazone,
                'Metformin-pioglitazone': metformin_pioglitazone
            }
            
            for field_name, field_value in medication_fields.items():
                if field_value is None or field_value == "":
                    missing_fields.append(field_name)
            
            # If there are missing fields, show error message
            if missing_fields:
                st.error(f"❌ The following required fields are missing or incomplete:")
                for field in missing_fields:
                    st.error(f"   • {field}")
                st.warning("⚠️ Please fill in all required fields before submitting.")
                return None
            
            # If all fields are filled, create the input DataFrame
            input_data = pd.DataFrame({
                'Race': [race],
                'Gender': [gender],
                'Age': [age],
                'time_in_hospital': [int(time_in_hospital)],
                'Speciality': [speciality],
                'max_glu_serum': [max_glu_serum],
                'A1Cresult': [a1cresult],
                'Metformin': [metformin],
                'Repaglinide': [repaglinide],
                'Nateglinide': [nateglinide],
                'Chlorpropamide': [chlorpropamide],
                'Glimepiride': [glimepiride],
                'Acetohexamide': [acetohexamide],
                'Glipizide': [glipizide],
                'Glyburide': [glyburide],
                'Tolbutamide': [tolbutamide],
                'Pioglitazone': [pioglitazone],
                'Rosiglitazone': [rosiglitazone],
                'Acarbose': [acarbose],
                'Miglitol': [miglitol],
                'Tolazamide': [tolazamide],
                'Examide': [examide],
                'Citoglipton': [citoglipton],
                'Insulin': [insulin],
                'Glyburide-metformin': [glyburide_metformin],
                'glipizide-metformin': [glipizide_metformin],
                'Glimepiride-pioglitazone': [glimepiride_pioglitazone],
                'metformin-rosiglitazone': [metformin_rosiglitazone],
                'Metformin-pioglitazone': [metformin_pioglitazone],
                'Troglitazone': [troglitazone],
                'DiabetesMed': [diabetesmed]
            })
            
            st.success("✅ All required fields are complete!")
            return input_data
        return None

def show_prediction_result(prediction, probabilities):
    outcome = prediction[0]
    probability = probabilities[0][1] if outcome == "Yes" else probabilities[0][0]
    color = OUTCOME_COLORS[outcome]
    emoji = OUTCOME_EMOJIS[outcome]
    
    st.markdown(f"""
        <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center; 
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 20px;'>
            <h2 style='color: #2c3e50;'>Predicted Readmission Outcome</h2>
            <div style='font-size: 2.5rem; color: {color}; font-weight: bold;'>
                {emoji} {outcome} ({probability*100:.1f}%)
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.subheader("📋 Clinical Recommendations")
    if outcome == "No":
        st.success("""
        ✅ **Standard Discharge Plan:**
        - Standard discharge procedures
        - Schedule follow-up in 3 months
        - Continue current medication regimen
        - Provide standard diabetes education materials
        - Low risk intervention needed
        """)
    else:
        st.error("""
        ⚠️ **High Risk of Readmission Detected - Enhanced Care Plan Required:**
        - Enhanced discharge planning required
        - Schedule follow-up within 2 weeks
        - Comprehensive medication review
        - Intensive diabetes education
        - Social work and nutrition consults recommended
        - Consider home health referral
        - Patient may benefit from transition care program
        """)

def display_dataframe(df):
    """Optimized DataFrame display function"""
    if len(df) > 1000:
        st.warning(f"Showing first 1000 rows out of {len(df)}")
        df_display = df.head(1000)
    else:
        df_display = df.copy()
    st.dataframe(df_display.astype(str))

def risk_prediction_page():
    st.title("🏥 Diabetic Patient Readmission Prediction")
    st.markdown(f"Welcome back, **{st.session_state.user_info['username']}** ({st.session_state.user_info['role']}) from **{st.session_state.user_info['hospital']}**")
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["🩺 Single Patient", "📊 Batch Analysis"])
    
    with tab1:
        st.header("Individual Patient Assessment")
        input_data = get_input_features(data)
        
        if input_data is not None:
            with st.spinner('🔍 Analyzing patient data...'):
                try:
                    prediction = model.predict(input_data)
                    probabilities = model.predict_proba(input_data)
                    show_prediction_result(prediction, probabilities)
                    
                    st.subheader("📊 Probability Breakdown")
                    prob_df = pd.DataFrame({
                        'Outcome': ['No Readmission', 'Readmission'],
                        'Probability': [probabilities[0][0]*100, probabilities[0][1]*100]
                    })
                    fig = px.bar(prob_df, x='Outcome', y='Probability', 
                                color='Outcome', 
                                color_discrete_map={'No Readmission': '#2ecc71', 'Readmission': '#e74c3c'},
                                text='Probability', 
                                title='Readmission Probability Distribution')
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("📄 Patient Summary")
                    display_dataframe(input_data)
                    
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")
                    st.info("Please check that all input values are valid and try again.")
    
    with tab2:
        st.header("Batch Patient Analysis")
        st.info("Upload a CSV or Excel file with patient data for batch predictions. All required fields must be present.")
        
        uploaded_file = st.file_uploader("Choose an Excel or CSV file", 
                                       type=["xlsx", "csv"],
                                       accept_multiple_files=False)
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                else:
                    df = pd.read_csv(uploaded_file)
                
                st.success("✅ File successfully uploaded!")
                st.write("📋 Preview of uploaded data:")
                display_dataframe(df)
                
                # Validate required columns
                required_columns = ['Race', 'Gender', 'Age', 'time_in_hospital', 'Speciality', 
                                  'max_glu_serum', 'A1Cresult', 'Metformin', 'Insulin', 'DiabetesMed']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                    st.info("Please ensure your file contains all required columns.")
                else:
                    if st.button("🚀 Run Batch Analysis", type="primary"):
                        with st.spinner('🔄 Processing patient records...'):
                            try:
                                predictions = model.predict(df)
                                probabilities = model.predict_proba(df)
                                
                                results = df.copy()
                                results["Predicted_Outcome"] = predictions
                                results["Readmission_Probability"] = [prob[1] for prob in probabilities]
                                
                                st.session_state.results = results
                                st.success("✅ Batch analysis completed successfully!")
                            except Exception as e:
                                st.error(f"❌ Error processing batch: {str(e)}")
            
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.info("Please ensure your file is properly formatted.")
        
        if "results" in st.session_state:
            st.subheader("📈 Readmission Prediction Results")
            display_dataframe(st.session_state.results)
            
            # Download button
            csv = st.session_state.results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv,
                file_name="patient_readmission_predictions.csv",
                mime="text/csv"
            )
            
            st.subheader("📊 Prediction Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = px.pie(st.session_state.results, 
                             names="Predicted_Outcome",
                             title="Readmission Distribution",
                             color="Predicted_Outcome",
                             color_discrete_map=OUTCOME_COLORS)
                fig1.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                category = st.selectbox("Group by", 
                                      options=["Race", "Gender", "Age", "Speciality"],
                                      index=0)
                
                fig2 = px.bar(st.session_state.results.groupby([category, "Predicted_Outcome"])
                             .size()
                             .unstack(),
                             title=f"Readmissions by {category}",
                             color_discrete_map=OUTCOME_COLORS,
                             labels={'value': 'Count', 'variable': 'Outcome'})
                fig2.update_layout(barmode='group', height=400)
                st.plotly_chart(fig2, use_container_width=True)
    
def login_page():
    st.header("🔐 Login to Readmission Prediction System")
    with st.form("login_form"):
        username = st.text_input("Username *", placeholder="Enter your username")
        password = st.text_input("Password *", type="password", placeholder="Enter your password")
        submitted = st.form_submit_button("Login", type="primary")
        
        if submitted:
            if not username or not password:
                st.warning("⚠️ Please fill in all fields")
                return
            
            success, user = login_user(username, password)
            if success:
                st.session_state.authenticated = True
                st.session_state.user_info = user
                st.success("✅ Login successful!")
                st.balloons()
                st.rerun()
            else:
                st.error("❌ Invalid username or password")

def registration_page():
    st.title("👤 New User Registration")
    st.markdown("---")
    st.markdown("**⚠️ All fields marked with * are required**")
    
    with st.form("Register"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Account Information")
            username = st.text_input("Username *", help="Required field")
            password = st.text_input("Password *", type="password", help="Required field")
            confirm_password = st.text_input("Confirm Password *", type="password", help="Required field")
            email = st.text_input("Email *", help="Required field")
            
        with col2:
            st.subheader("Professional Information")
            hospital = st.text_input("Hospital/Clinic Name *", help="Required field")
            role = st.selectbox("Role *", 
                              ["Doctor", "Nurse", "Administrator", "Researcher", "Other"],
                              index=None,
                              placeholder="Select your role...",
                              help="Required field")
            
        st.markdown("---")
        submitted = st.form_submit_button("Register", type="primary")
        
        if submitted:
            missing_fields = []
            if not username:
                missing_fields.append("Username")
            if not password:
                missing_fields.append("Password")
            if not confirm_password:
                missing_fields.append("Confirm Password")
            if not email:
                missing_fields.append("Email")
            if not hospital:
                missing_fields.append("Hospital/Clinic Name")
            if not role:
                missing_fields.append("Role")
            
            if missing_fields:
                st.error("❌ The following required fields are missing:")
                for field in missing_fields:
                    st.error(f"   • {field}")
            elif password != confirm_password:
                st.error("❌ Passwords do not match")
            elif not is_valid_email(email):
                st.error("❌ Invalid email format. Please enter a valid address, e.g. name@example.com")
            else:
                success, message = register_user(username, password, email, hospital, role)
                if success:
                    st.success("✅ " + message)
                    st.info("🔐 Please login with your new credentials")
                else:
                    st.error("❌ " + message)

def main():
    if not st.session_state.authenticated:
        tab1, tab2 = st.tabs(["🔐 Login", "👤 Register"])
        with tab1:
            login_page()
        with tab2:
            registration_page()
    else:
        risk_prediction_page()
        
        st.sidebar.title("👤 Account Info")
        st.sidebar.markdown(f"""
            **Username:** {st.session_state.user_info['username']}  
            **Hospital:** {st.session_state.user_info['hospital']}  
            **Role:** {st.session_state.user_info['role']}
        """)
        
        st.sidebar.markdown("---")
        st.sidebar.info("""
            **ℹ️ About this App:**  
            This application predicts hospital readmission for diabetic patients using machine learning.
            Outcomes are binary (Yes/No) with probability scores.
            
            **📋 Required Fields:**  
            All fields are required for prediction.
        """)
        
        if st.sidebar.button("🚪 Logout", type="primary"):
            st.session_state.authenticated = False
            st.session_state.user_info = None
            st.rerun()

if __name__ == "__main__":
    main()