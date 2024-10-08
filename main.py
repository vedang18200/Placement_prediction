import pandas as pd
import joblib


salary_model = 'model/salary/salary_model.pkl'
salary_scaler = 'model/salary/scaler_salary.pkl'
placement_model = 'model/placement/svm_model.pkl'
placement_scaler = 'model/placement/scaler.pkl'


salary_regressor = joblib.load(salary_model)
salary_scaler = joblib.load(salary_scaler)


placement_clf = joblib.load(placement_model)
placement_sc = joblib.load(placement_scaler)

def predict_salary(features):
    columns = ['ssc_p', 'hsc_p', 'degree_p', 'internship', 'aptitiude_t']
    new_data = pd.DataFrame([features], columns=columns)
    new_data_scaled = salary_scaler.transform(new_data)
    salary_pred = salary_regressor.predict(new_data_scaled)
    return salary_pred[0]


def predict_placement_probability(features):
    columns_order = ['CGPA', 'Internships', 'Projects', 'Workshops/Certifications',
                     'AptitudeTestScore', 'SoftSkillsRating', 'ExtracurricularActivities',
                     'PlacementTraining', 'SSC_Marks', 'HSC_Marks']
    new_data = pd.DataFrame([features], columns=columns_order)
    new_data_scaled = placement_sc.transform(new_data)
    probabilities = placement_clf.predict_proba(new_data_scaled)
    return probabilities[0][1]

# Function to store user input in a CSV file
def store_user_input(data, filename='user_data/user_inputdata.csv'):
    df = pd.DataFrame([data], columns=[
        'SSC_Marks', 'HSC_Marks', 'Degree_CGPA', 'AptitudeTestScore', 'Projects',
        'Workshops', 'SoftSkillsRating', 'ExtracurricularActivities', 'PlacementTraining',
        'Internships'
    ])
    df.to_csv(filename, mode='a', header=not pd.io.common.file_exists(filename), index=False)

# Function to get user input
def get_user_input():
    SSC_Marks = float(input("SSC Percentage: "))
    HSC_Marks = float(input("HSC Percentage: "))
    CGPA = float(input("Degree CGPA: "))
    AptitudeTestScore = float(input("Aptitude Test Score: "))
    Projects = float(input("Number of Projects: "))
    Workshops = float(input("Number of Workshops/Certifications done: "))
    SoftSkillsRating = float(input("SoftSkills Rating out of 5: "))
    ExtracurricularActivities = float(input("Extra Curricular Activities (1 or 0): "))
    PlacementTraining = float(input("Placement Training (1 or 0): "))
    
    # Calculate additional features
    percentage = 7.1 * CGPA + 11
    Internships_input = float(input("Number of Internships: "))
    Internships = 1 if Internships_input > 0 else 0

    # Return features for both predictions
    salary_features = [SSC_Marks, HSC_Marks, percentage, Internships, AptitudeTestScore]
    placement_features = [CGPA, Internships, Projects, Workshops, AptitudeTestScore,
                          SoftSkillsRating, ExtracurricularActivities, PlacementTraining,
                          SSC_Marks, HSC_Marks]
    
    # Store user input data in CSV file
    user_input_data = [
        SSC_Marks, HSC_Marks, CGPA, AptitudeTestScore, Projects, Workshops, 
        SoftSkillsRating, ExtracurricularActivities, PlacementTraining, Internships
    ]
    store_user_input(user_input_data)
    
    return salary_features, placement_features

# Get user input
salary_features, placement_features = get_user_input()

# Predict the salary
predicted_salary = predict_salary(salary_features)
print(f"Predicted Salary: {predicted_salary:.2f} LPA")

# Predict the placement probability
placement_probability = predict_placement_probability(placement_features)
print(f"Probability of being placed: {placement_probability * 100:.2f}%")
