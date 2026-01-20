import streamlit as st 
import requests
import os 
from us_visa.constants.constant import APP_PORT
from dotenv import load_dotenv

load_dotenv()

public_ip = os.getenv("EC2_PUBLIC_IP","localhost")
API_URL = f"http://{public_ip}:{APP_PORT}/predict"

st.title("US Visa Prediction System")

with st.form("visa_form"):
    st.header("Visa Application Form")

    continent =  st.selectbox(label="Continent",
                              options=["Asia","Africa","North America","Europe","South America","Oceania"])
    
    education_of_employee = st.selectbox("Education of Employee",
                                         options=["High School","Bachelor's","Master's","Doctorate"],
                                         help="Select the highest level of education")
    
    has_job_experience = st.selectbox("Has Job Experience",
                                      options=["Y","N"],
                                      help="Does the employee have any job experience ?")
    
    requires_job_training = st.selectbox("Requires Job Training",
                                         options=["Y","N"],
                                         help="Does the job require any training ?")
    
    no_of_employees = st.number_input("Number of Employees",
                                      min_value=14500,
                                      max_value=40000,
                                      value=20000,
                                      step=500,
                                      help="Enter the number of employees in the company")
    
    region_of_employment = st.selectbox("Region of Employment",
                                        options=["West","Northeast","South","Midwest","Island"],
                                        help="Select the employment region")
    
    prevailing_wage = st.number_input("Prevailing Wage",
                                      min_value=600.0,
                                      max_value=100000.0,
                                      value=10000.0,
                                      step=500.0,
                                      help="Enter the prevailing wage")
    
    unit_of_wage = st.selectbox("Prevailing Wage",
                                options=["Hour","Year","Week","Month"],
                                help="Select the contract tenure unit")
    
    full_time_position = st.selectbox("Full Time Position",
                                      options=["Y","N"],
                                      help="Is it a full time position ?")
    
    company_age = st.number_input("Age of the company",
                                  min_value=15,
                                  max_value=180,
                                  value=20,
                                  step=5,
                                  help="Enter the age of the company")
    
    submit_button = st.form_submit_button(label="Predict")

if submit_button:
    # Prepare the data payload
    payload = {
        "continent": continent,
        "education_of_employee": education_of_employee,
        "has_job_experience": has_job_experience,
        "requires_job_training": requires_job_training,
        "no_of_employees": no_of_employees,
        "region_of_employment": region_of_employment,
        "prevailing_wage": prevailing_wage,
        "unit_of_wage": unit_of_wage,
        "full_time_position": full_time_position,
        "company_age": company_age,
    }

    try:
        # Make the POST request to the FastAPI endpoint
        response = requests.post(API_URL, json=payload)
        response_data = response.json()

        if response.status_code == 200 and response_data.get("status", False):
            st.success(f"Prediction: {response_data['prediction']}")
        else:
            st.error(f"Error: {response_data.get('error', 'Unknown error')}")
    except Exception as e:
        st.error(f"Error connecting to the API: {str(e)}")