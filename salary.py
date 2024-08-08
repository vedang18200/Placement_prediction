import pandas as pd
import joblib

salary_model = 'model/salary/salary_model.pkl'
salary_scaler = 'model/salary/scaler_salary.pkl'

regressor = joblib.load(salary_model)
scaler = joblib.load(salary_scaler)

# Function to predict salary
def predict_salary(features):
    
    columns = ['ssc_p', 'hsc_p', 'degree_p', 'internship', 'aptitiude_t']
    new_data = pd.DataFrame([features], columns=columns)
    new_data_scaled = scaler.transform(new_data)
    salary_pred = regressor.predict(new_data_scaled)
    return salary_pred[0]

# Function to get user input
def get_user_input():
    
    SSC_Marks = float(input("SSC Percentage: "))
    HSC_Marks = float(input("HSC Percentage: "))
    CGPA = float(input("Degree CGPA: "))
    AptitudeTestScore = float(input("Aptitude Test Score: "))

    
    percentage = 7.1 * CGPA + 11

   
    Internships_input = float(input("Number of Internships: "))
    Internships = 1 if Internships_input > 0 else 0

    
    return [SSC_Marks, HSC_Marks, percentage, Internships, AptitudeTestScore]

# Get user input
new_features = get_user_input()

# Predict the salary
predicted_salary = predict_salary(new_features)
print(f"Predicted Salary: {predicted_salary:.2f} LPA")
