from django.shortcuts import render
import pandas as pd
import joblib

# Load your pre-trained models and scalers
salary_model = joblib.load('model/salary/salary_model.pkl')
salary_scaler = joblib.load('model/salary/scaler_salary.pkl')
placement_model = joblib.load('model/placement/svm_model.pkl')
placement_scaler = joblib.load('model/placement/scaler.pkl')

# Function to predict salary
def predict_salary(features):
    columns = ['ssc_p', 'hsc_p', 'degree_p', 'internship', 'aptitiude_t']
    new_data = pd.DataFrame([features], columns=columns)
    new_data_scaled = salary_scaler.transform(new_data)
    salary_pred = salary_model.predict(new_data_scaled)
    return salary_pred[0]

# Function to predict placement probability
def predict_placement_probability(features):
    columns_order = ['CGPA', 'Internships', 'Projects', 'Workshops/Certifications',
                     'AptitudeTestScore', 'SoftSkillsRating', 'ExtracurricularActivities',
                     'PlacementTraining', 'SSC_Marks', 'HSC_Marks']
    new_data = pd.DataFrame([features], columns=columns_order)
    new_data_scaled = placement_scaler.transform(new_data)
    probabilities = placement_model.predict_proba(new_data_scaled)
    return probabilities[0][1]

# Render the index.html for form input
def index(request):
    return render(request, 'index.html')

# Process form input and return predictions
def predict(request):
    if request.method == 'POST':
        # Get form data
        data = request.POST
        
        # Extract features from the form input
        ssc_p = float(data.get('ssc'))
        hsc_p = float(data.get('hsc'))
        degree_cgpa = float(data.get('cgpa'))
        aptitude_score = float(data.get('aptitude'))
        projects = float(data.get('projects'))
        workshops = float(data.get('workshops'))
        softskills_rating = float(data.get('softskills'))
        extracurricular = int(data.get('extra'))
        placement_training = int(data.get('training'))
        internships_input = int(data.get('internships'))

        # Calculate additional feature: percentage (based on CGPA)
        percentage = 7.1 * degree_cgpa + 11
        internships = 1 if internships_input > 0 else 0

        # Prepare features for salary and placement prediction
        salary_features = [ssc_p, hsc_p, percentage, internships, aptitude_score]
        placement_features = [
            degree_cgpa, internships, projects, workshops, aptitude_score, softskills_rating,
            extracurricular, placement_training, ssc_p, hsc_p
        ]
        
        # Predict salary and placement probability
        predicted_salary = predict_salary(salary_features)
        placement_probability = predict_placement_probability(placement_features)

        # Render the results on the same page (index.html)
        return render(request, 'index.html', {
            'salary': round(predicted_salary, 2),
            'placement_probability': round(placement_probability * 100, 2),
        })
    
    # If GET request, just render the form
    return render(request, 'index.html')
