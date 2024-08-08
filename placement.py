import pandas as pd
import joblib

model = 'model/placement/svm_model.pkl'
scaler = 'model/placement/scaler.pkl'

clf = joblib.load(model)
sc = joblib.load(scaler)


print("Please enter the following details:")


SSC_Marks = float(input("SSC Percentage: "))
HSC_Marks= float(input("HSC Percentage: "))
CGPA = float(input("Degree CGPA: "))
Internships= int(input("Number of Internship: "))
Projects = float(input("Number of Projects: "))
Workshops = float(input("Number of Workshops/Certifications done:"))
AptitudeTestScore = float(input("Aptitude Test Score:"))
SoftSkillsRating = float(input("SoftSkills Rating outof 5:"))
ExtracurricularActivities = float(input("Extra Curricular Activities (1 or 0):"))
PlacementTraining = float(input("Placement Training (1 or 0):"))




data = {
    'CGPA' : [CGPA],
    'Internships': [Internships],
    'Projects': [Projects],
    'Workshops/Certifications': [Workshops],
    'AptitudeTestScore': [AptitudeTestScore],
    'SoftSkillsRating':[SoftSkillsRating],
    'ExtracurricularActivities':[ExtracurricularActivities],
    'PlacementTraining':[PlacementTraining],
    'SSC_Marks':[SSC_Marks],
    'HSC_Marks':[HSC_Marks]
}

new_data = pd.DataFrame(data)


columns_order = ['CGPA','Internships','Projects','Workshops/Certifications','AptitudeTestScore','SoftSkillsRating','ExtracurricularActivities','PlacementTraining','SSC_Marks','HSC_Marks']
new_data = new_data[columns_order]


x_new_scaled = sc.transform(new_data)


probabilities = clf.predict_proba(x_new_scaled)


placement_probability = probabilities[0][1]

print(f"Probability of being placed: {placement_probability * 100:.2f}%")
