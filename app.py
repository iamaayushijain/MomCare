import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as gen_ai
import firebase_admin
from firebase_admin import credentials, auth, firestore
from streamlit_option_menu import option_menu
import pyrebase
import json
import matplotlib.pyplot as plt
import numpy as np
import joblib
import pandas as pd
import seaborn as sns
model = joblib.load('risk_model.pkl')
# Define the prediction function
def predict_risk(age, systolic_bp, diastolic_bp, bs, heart_rate, body_temp):
    # Prepare the input as a DataFrame (assuming the model expects a DataFrame input)
    input_data = pd.DataFrame([[age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate,]],
                              columns=['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate'])
    
    # Predict risk level
    risk_level = model.predict(input_data)[0]
    return risk_level
def plot_feature_importance(model):
    feature_importance = model.feature_importances_
    features = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'HeartRate', 'BodyTemp']
    feature_score = pd.Series(feature_importance, index=features)
    score = feature_score.sort_values()
    
    plt.figure(figsize=(8, 4))
    plt.barh(y=score.index, width=score.values, color='violet')
    plt.grid(alpha=0.4)
    plt.title('Feature Importance of Random Forest')
    st.pyplot(plt)
# Load environment variables
load_dotenv()

firebase_config = {
    "apiKey": os.getenv("FIREBASE_API_KEY"),
    "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN"),
    "projectId": os.getenv("FIREBASE_PROJECT_ID"),
    "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
    "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID"),
    "appId": os.getenv("FIREBASE_APP_ID"),
    "measurementId": os.getenv("FIREBASE_MEASUREMENT_ID"),
    "type": "service_account",
    "project_id": os.getenv("FIREBASE_PROJECT_ID"),
    "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
    "private_key": os.getenv("FIREBASE_PRIVATE_KEY").replace('\\n', '\n'),  # Ensure newlines are correctly handled
    "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
    "client_id": os.getenv("FIREBASE_CLIENT_ID"),
    "auth_uri": os.getenv("FIREBASE_AUTH_URI"),
    "token_uri": os.getenv("FIREBASE_TOKEN_URI"),
    "auth_provider_x509_cert_url": os.getenv("FIREBASE_AUTH_PROVIDER_X509_CERT_URL"),
    "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_X509_CERT_URL"),
    "universe_domain": os.getenv("FIREBASE_UNIVERSE_DOMAIN"),
    "databaseURL": os.getenv("FIREBASE_DATABASE_URL")
}



# Initialize Firebase Admin SDK only once
if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_config)
    firebase_admin.initialize_app(cred)
# Firestore Database
db = firestore.client()
# Firebase auth


firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()

# Configure Streamlit page settings
st.set_page_config(
    page_title="Chat with Gemini-Pro!",
    page_icon=":brain:",  # Favicon emoji
    layout="centered",  # Page layout option
)
# Function for user login and signup
import streamlit as st
from firebase_admin import firestore  # Ensure Firebase is initialized and Firestore is imported

from streamlit_option_menu import option_menu
# LOGIN CODE ###############################
def login_signup():
    choice = st.selectbox("Login/Signup", ["Login", "Signup"])
    email = st.text_input("Enter your email")
    password = st.text_input("Enter your password", type="password")
    # Signup Logic
    if choice == "Signup":
        signup_triggered = st.button("Create Account")
        if signup_triggered and email and password:
            try:
                user = auth.create_user_with_email_and_password(email=email, password=password)
                st.success("Account created successfully!")
            except Exception as e:
                st.error(f"Error creating account: {e}")
                print(e)  # Debug statement
        elif signup_triggered:
            st.error("Please enter both email and password.")
    # Login Logic
    if choice == "Login":
        login_triggered = st.button("Login")
        if login_triggered and email and password:
            try:
                # Use Firebase Authentication API to sign in
                user = auth.sign_in_with_email_and_password(email, password)
                st.session_state.user = user
                st.session_state.logged_in = True  # Set login flag to True
                st.session_state.user_id = user['localId']  # Store user ID for later use
                st.success("Logged in successfully!")
                st.rerun()  # Rerun the script to show the main app
            except Exception as e:
                st.error(f"Invalid credentials: {e}")
                print(e)  # Debug statement
        elif login_triggered:
            st.error("Please enter both email and password.")

# If user is not logged in, show login/signup page
if 'logged_in' not in st.session_state or not st.session_state.logged_in:
    login_signup()
else:
    user_id = st.session_state.user_id  # Ensure user ID is available for later use
    # Sidebar Navigation
    with st.sidebar:
        selected = option_menu(
            menu_title="Menu",
            options=["Chatbot", "Prediction", "Profile"],
            icons=["house-heart-fill", "calendar2-heart-fill", "envelope-heart-fill"],
            menu_icon="heart-eyes-fill",
            default_index=0,
        )
    # Firestore Profile Retrieval (with cache clearing for updates)
    @st.cache_data
    def get_profile_data(user_id):
        profile_doc = db.collection("users").document(user_id).get()
        print("hello1")
        if profile_doc.exists:
            print("hello2")
            return profile_doc.to_dict()
        return {}  # Return empty dictionary if no data exists
    # Profile Section ##########################################################
    if selected == "Profile":
        col1, col2 = st.columns([7, 1])
        with col1:
            st.subheader("Build Your Profile")
        with col2:
            if st.button("Logout"):
                del st.session_state.user
                del st.session_state.logged_in
                st.rerun()
        # Load existing profile data from Firestore
        print("hello3")
        profile_data = get_profile_data(user_id)
        # Pre-fill the form with existing data or defaults
        name = st.text_input("Name", value=profile_data.get("name", ""))
        age = st.text_input("Age", value=str(profile_data.get("age", "")))  # Convert to str
        months_pregnant = st.text_input("Months Pregnant", value=str(profile_data.get("months_pregnant", "")))  # Convert to str
        chronic_diseases = st.text_area("Chronic Diseases (separate by commas)", 
                                        value=', '.join(profile_data.get("chronic_diseases", [])))
        weight = st.text_input("Weight (kg)", value= profile_data.get("weight", ""))  # Convert to str
        height = st.text_input("Height (cm)", value= profile_data.get("height", ""))  # Convert to str
        medications = st.text_area("Medications", value=', '.join(profile_data.get("medications", [])))
        allergies = st.text_area("Allergies", value=', '.join(profile_data.get("allergies", [])))
        exercise = st.text_input("Exercise Routine", value=profile_data.get("exercise", ""))
        dietary_preferences = st.text_input("Dietary Preferences", value=profile_data.get("dietary_preferences", ""))
        smoking_habits = st.text_input("Smoking Habits", value=profile_data.get("smoking_habits", ""))
        alcohol_habits = st.text_input("Alcohol Consumption", value=profile_data.get("alcohol_habits", ""))
        if height and weight:
            weightf = float(weight)
            heightf = float(height) / 100
            bmi = weightf / (heightf ** 2)
        
            st.markdown(f"<p style='color:red; font-size:20px;'>  Your BMI is {bmi:.2f} </p>", unsafe_allow_html=True)
            if bmi < 18.5:
                st.markdown("<p style='color:blue; font-size:18px;'>Your BMI is considered Underweight.</p>", unsafe_allow_html=True)
            elif 18.5 <= bmi <= 24.9:
                st.markdown("<p style='color:green; font-size:18px;'>Your BMI is in the Good (Normal) range.</p>", unsafe_allow_html=True)
            elif 25 <= bmi <= 29.9:
                st.markdown("<p style='color:orange; font-size:18px;'>Your BMI is considered Okay (Overweight).</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color:red; font-size:18px;'>Your BMI is in the Bad (Obese) range.</p>", unsafe_allow_html=True)
        # Save profile data to Firestore on button click
        if st.button("Save Profile"):
            # Update profile data and reformat lists correctly
            updated_profile_data = {
                "name": name,
                "age": int(age) if age.isdigit() else age,
                "months_pregnant": int(months_pregnant) if months_pregnant.isdigit() else months_pregnant,
                "chronic_diseases": [disease.strip() for disease in chronic_diseases.split(',') if disease.strip()],
                "weight": float(weight) if weight.replace('.', '', 1).isdigit() else weight,
                "height": float(height) if height.replace('.', '', 1).isdigit() else height,
                "medications": [med.strip() for med in medications.split(',') if med.strip()],
                "allergies": [allergy.strip() for allergy in allergies.split(',') if allergy.strip()],
                "exercise": exercise,
                "dietary_preferences": dietary_preferences,
                "smoking_habits": smoking_habits,
                "alcohol_habits": alcohol_habits
            }
            # Save to Firestore
            db.collection("users").document(user_id).set(updated_profile_data)
            st.success("Profile saved successfully!")
            # Clear the cache to fetch updated data on next load
            get_profile_data.clear()
 ###############################
    # Chatbot Section
    # Chatbot Section
    elif selected == "Chatbot":
        # Load API key from environment variables
        GOOGLE_API_KEY = os.getenv("API_Key")
        # Set up Google Gemini-Pro AI model
        gen_ai.configure(api_key=GOOGLE_API_KEY)
        model = gen_ai.GenerativeModel('gemini-pro')
        # Function to translate roles between Gemini-Pro and Streamlit terminology
        def translate_role_for_streamlit(user_role):
            return "assistant" if user_role == "model" else user_role
        # Initialize chat session and display history in Streamlit
        if "chat_session" not in st.session_state:
            st.session_state.chat_session = model.start_chat(history=[])
            st.session_state.display_history = []  # New: Store only user prompt and assistant response here
        
        if "display_history" not in st.session_state:
            st.session_state.display_history = []  # Stores only user prompts and assistant responses
    
        # Fetch profile data
        profile_data = get_profile_data(user_id)
        # Function to get profile data as a formatted string
        def get_profile_info():
            return "\n".join([
                f"Name: {profile_data.get('name', 'N/A')}",
                f"Age: {profile_data.get('age', 'N/A')}",
                f"Months Pregnant: {profile_data.get('months_pregnant', 'N/A')}",
                f"Chronic Diseases: {', '.join(profile_data.get('chronic_diseases', []))}",
                f"Weight: {profile_data.get('weight', 'N/A')} kg",
                f"Height: {profile_data.get('height', 'N/A')} cm",
                f"Medications: {', '.join(profile_data.get('medications', []))}",
                f"Allergies: {', '.join(profile_data.get('allergies', []))}",
                f"Exercise Routine: {profile_data.get('exercise', 'N/A')}",
                f"Dietary Preferences: {profile_data.get('dietary_preferences', 'N/A')}",
                f"Smoking Habits: {profile_data.get('smoking_habits', 'N/A')}",
                f"Alcohol Consumption: {profile_data.get('alcohol_habits', 'N/A')}"
            ])
        # Chatbot UI
        st.title("🤖 Your AI Health Assistant")
        languages = [
            "Hindi", "Bengali", "Telugu", "Marathi", "Tamil", 
            "Urdu", "Gujarati", "Malayalam", "Kannada", "Odia", 
            "Punjabi", "Assamese", "Maithili", "Sanskrit", 
            "Konkani", "Sindhi", "Nepali", "Kashmiri", "Dogri", 
            "Santali", "Bodo"
        ]
        # Dropdown list to select language
        selected_language = st.selectbox("Select a language", languages)
        ## HISTORY DISPLAY
        for message in st.session_state.display_history:
            if message["role"] == "user":
                st.chat_message("user").markdown(message["text"])  # Display user prompt
            else:
                st.chat_message("assistant").markdown(message["text"])  # Display assistant response
        # Get user input
        user_prompt = st.chat_input("Ask Gemini-Pro...")
        if user_prompt:
            # Prepare full prompt with profile data for the model
            complete_prompt = f"{get_profile_info()}\n\nUser Query: {user_prompt} \n\nLanguage we want the answer in: {selected_language}  "
            # FIRST USER DISPLAY 
            
            # Display user's question in the chat
            with st.chat_message("user"):
                st.markdown(user_prompt)
            ####
            
            # Send prompt to Gemini-Pro
            response = st.session_state.chat_session.send_message(complete_prompt)
            
            # Store user prompt and response for display only
            st.session_state.display_history.append({"role": "user", "text": user_prompt})
            st.session_state.display_history.append({"role": "assistant", "text": response.text})
            
            ### FIRST ASSISTANT DISPlAY 
            # Display assistant’s response
            with st.chat_message("assistant"):
                st.markdown(response.text)
            ###
# PREDICTION PAGE ____________________________________________________________________
    elif selected == "Prediction":
           # Page Title
        st.title("Prediction")
        
     
        # Input Section
        st.header("Input Patient Data")
    
        age = st.slider("Age:", min_value=30, max_value=100, value=50, step=5)
        systolic_bp = st.slider("Systolic BP (mmHg):", min_value=90, max_value=200, value=120, step=1)
        diastolic_bp = st.slider("Diastolic BP (mmHg):", min_value=60, max_value=120, value=80, step=1)
        bs = st.slider("Blood Glucose Levels (mmol/L):", min_value=3.0, max_value=15.0, value=5.0, step=0.1)
        heart_rate = st.slider("Heart Rate (bpm):", min_value=40, max_value=150, value=70, step=1)
        body_temp = st.slider("Body Temperature (°F):", min_value=35.0, max_value=42.0, value=37.0, step=0.1)
        # Button to Trigger Prediction
        if st.button("Predict Risk Level"):
            risk_level = predict_risk(age, systolic_bp, diastolic_bp, bs, heart_rate, body_temp)
            
            # Displaying the Risk Level
            st.success(f"Predicted Risk Level: {risk_level}")
            
            # Plot the graph
            st.subheader("Feature Importance in Model Prediction")
            plot_feature_importance(model)