from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from functools import wraps
import joblib
import pickle
import pandas as pd
import numpy as np
import google.generativeai as genai
import os
from chat import get_health_recommendation,fetch_dataset,ask_gemini,healthcare_chatbot
from flask import jsonify, session
from Models import db
from Models import  User, ChatLog

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///Usersss.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.secret_key = "your_secret_key"




def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['name']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])

        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'danger')
            return redirect(url_for('signup'))
        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'danger')
            return redirect(url_for('signup'))

        new_user = User(username=username, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()

        flash('Account created! Please login.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        login_input = request.form['username']
        password = request.form['password']

        user = User.query.filter(
            (User.username == login_input) | (User.email == login_input)
        ).first()

        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Logged in successfully!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid credentials', 'danger')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


model = joblib.load("models/heart_disease_model.pkl")
scaler = joblib.load("models/heart_scaler.pkl")

lung_model = joblib.load("models/lung_cancer_model.pkl")
lung_scaler = joblib.load("models/lung_scaler.pkl")
lung_encoders = pickle.load(open("models/lung_label_encoders.pkl", "rb"))

liver_model = pickle.load(open("models/Liver_Cirrhosis_Stage_Detection_DT.pkl", "rb"))
liver_encoders = pickle.load(open("models/liver_label_encoders.pkl", "rb"))

feature_names = [
    "age", "sex", "chest pain type", "resting bp s", "cholesterol", 
    "fasting blood sugar", "resting ecg", "max heart rate", "exercise angina", 
    "oldpeak", "ST slope"
]


def preprocess_input(data):
    mapping = {
        "sex": {"male": 1, "female": 0},
        "chest pain type": {
            "typical angina": 1, "atypical angina": 2,
            "non-anginal pain": 3, "asymptomatic": 4
        },
        "fasting blood sugar": {"yes": 1, "no": 0},
        "resting ecg": {
            "normal": 0, "ST-T wave abnormality": 1, "left ventricular hypertrophy": 2
        },
        "exercise angina": {"yes": 1, "no": 0},
        "ST slope": {"upward": 1, "flat": 2, "downward": 3}
    }
    
  
    for key, value in mapping.items():
        if key in data:
            data[key] = value[data[key]]
    
    return data

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/history", methods=["GET"])
def get_history():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401

    history = ChatLog.query.filter_by(user_id=user_id).order_by(ChatLog.timestamp.desc()).limit(20).all()

    history_data = [
        {
            "user_message": chat.user_message,
            "bot_response": chat.bot_response,
            "timestamp": chat.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        }
        for chat in history
    ]

    return jsonify({"history": history_data})

@app.route("/chatbot", methods=["POST"])
def chatbot():
    user_id = session.get("user_id")  
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    user_input = data.get("message")
    
 
    response = healthcare_chatbot(user_input)

 
    try:
        new_log = ChatLog(
            user_id=user_id,
            user_message=user_input,
            bot_response=response
        )
        db.session.add(new_log)
        db.session.commit()
    except Exception as e:
        print("❌ Failed to save chat log:", e)
        db.session.rollback()

    return jsonify({"response": response})

@app.route("/services")
def chathome():
    return render_template("chatbot.html")


@app.route("/heart_diagnosis")
@login_required
def heart_diagnosis():
    return render_template("heart_input.html")
@app.route('/lung_diagnosis')
@login_required
def lung_diagnosis():
    return render_template('lungs_input.html')
@app.route('/liver_diagnosis')
@login_required
def liver_diagnosis():
    return render_template('liver_input.html')
'''-------------------------------------------------'''



@app.route("/predict", methods=["POST"])
@login_required
def predict():
    # Get user input from form
    input_data = {
        "age": int(request.form["age"]),
        "sex": request.form["sex"],
        "chest pain type": request.form["chest pain type"],
        "resting bp s": int(request.form["resting bp s"]),
        "cholesterol": int(request.form["cholesterol"]),
        "fasting blood sugar": request.form["fasting blood sugar"],
        "resting ecg": request.form["resting ecg"],
        "max heart rate": int(request.form["max heart rate"]),
        "exercise angina": request.form["exercise angina"],
        "oldpeak": float(request.form["oldpeak"]),
        "ST slope": request.form["ST slope"]
    }

    # Convert categorical values
    processed_data = preprocess_input(input_data)

    # Convert to DataFrame and scale
    input_df = pd.DataFrame([processed_data])
    input_scaled = scaler.transform(input_df)

    # Get prediction probability
    prediction_proba = model.predict_proba(input_scaled)[0][1]
    prediction_percent = round(prediction_proba * 100, 2)

    # Determine risk label
    result = f"{prediction_percent}% chance of Heart Disease"
    risk_level = "High Risk" if prediction_proba >= 0.5 else "Low Risk"

    
    try:
        feature_importance = model.feature_importances_
        feature_impact = {
            feature_names[i]: round(feature_importance[i] * 100, 2)
            for i in range(len(feature_names))
        }
    except AttributeError:
        feature_impact = {feature: "N/A (Model does not support feature importance)" for feature in feature_names}

    feature_impact_sorted = dict(sorted(feature_impact.items(), key=lambda item: item[1], reverse=True))

  
    recommendation = get_health_recommendation(input_data, result)

    return render_template(
        "heart_result.html",
        prediction=result,
        risk_level=risk_level,
        feature_impact=feature_impact_sorted,
        recommendation=recommendation
    )
@app.route('/predict_lung', methods=['POST'])
@login_required
def predict_lung():
    if request.method == 'POST':
        try:
            form_data = request.form

            age = float(form_data['age'])
            bmi = float(form_data['bmi'])
            cholesterol_level = float(form_data['cholesterol_level'])
            treatment_duration = float(form_data['treatment_duration'])

            categorical_features = [
                'gender', 'cancer_stage', 'smoking_status',
                'hypertension', 'asthma', 'cirrhosis',
                'other_cancer', 'treatment_type'
            ]

            encoded_features = []
            for feature in categorical_features:
                value = form_data[feature].strip().lower()
                if feature in lung_encoders:
                    if value in lung_encoders[feature].classes_:
                        encoded_value = lung_encoders[feature].transform([value])[0]
                    else:
                        encoded_value = lung_encoders[feature].transform(
                            [lung_encoders[feature].classes_[0]]
                        )[0]
                else:
                    encoded_value = 0
                encoded_features.append(encoded_value)

            numerical_features = np.array([[age, bmi, cholesterol_level, treatment_duration]])
            numerical_features = lung_scaler.transform(numerical_features)

            encoded_features = np.array(encoded_features).reshape(1, -1)
            final_features = np.hstack((numerical_features, encoded_features))

            prediction_proba = lung_model.predict_proba(final_features)[0][1]
            prediction_percentage = round(prediction_proba * 100, 2)
            result = "Survived" if prediction_proba >= 0.5 else "Not Survived"

            try:
                feature_importance = lung_model.feature_importances_
                feature_names = ['Age', 'BMI', 'Cholesterol Level', 'Treatment Duration'] + categorical_features
                feature_impact = {
                    feature_names[i]: round(feature_importance[i] * 100, 2)
                    for i in range(len(feature_names))
                }
                feature_impact_sorted = dict(sorted(feature_impact.items(), key=lambda item: item[1], reverse=True))
            except AttributeError:
                feature_impact_sorted = {feature: "N/A" for feature in feature_names}

            # Generate recommendations
            recommendations = []

            if prediction_proba < 0.5:
                recommendations.append("⚠️ Your survival likelihood is below 50%. Immediate medical attention and consistent follow-up is strongly advised.")
                if form_data['smoking_status'].lower() == 'yes':
                    recommendations.append("🚭 Consider joining a smoking cessation program.")
                if float(bmi) > 25:
                    recommendations.append("🥗 Adopt a balanced diet and engage in regular light physical activity.")
                if float(cholesterol_level) > 200:
                    recommendations.append("🫀 Reduce cholesterol through a low-fat diet and medication as prescribed.")
                if form_data['cancer_stage'].lower() in ['stage 3', 'stage 4']:
                    recommendations.append("🔬 Please consult an oncologist for advanced treatment planning.")
            else:
                recommendations.append("✅ Great! Your survival prediction is positive. Keep maintaining healthy habits.")
                if float(bmi) >= 25:
                    recommendations.append("💪 Try to bring BMI into the healthy range with activity and nutrition.")
                recommendations.append("🩺 Continue regular checkups to stay ahead of any issues.")
            
           
            recommendations.append("💡 Stay positive and proactive about your health.")

            return render_template(
                'lungs_result.html',
                prediction=result,
                prediction_percentage=prediction_percentage,
                feature_impact=feature_impact_sorted,
                recommendations=recommendations
            )

        except Exception as e:
            return f"Error: {str(e)}", 400

@app.route('/predict_liver', methods=['POST'])
@login_required
def predict_liver():
    try:
        
        inputs = {
            "Age": float(request.form['Age']),
            "Bilirubin": float(request.form['Bilirubin']),
            "Cholesterol": float(request.form['Cholesterol']),
            "Albumin": float(request.form['Albumin']),
            "Copper": float(request.form['Copper']),
            "Alk_Phos": float(request.form['Alk_Phos']),
            "SGOT": float(request.form['SGOT']),
            "Tryglicerides": float(request.form['Tryglicerides']),
            "Platelets": float(request.form['Platelets']),
            "Prothrombin": float(request.form['Prothrombin'])
        }

        # Encode categorical inputs
        categorical_features = ["Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema"]
        for feature in categorical_features:
            if request.form[feature] in liver_encoders[feature].classes_:
                inputs[feature] = liver_encoders[feature].transform([request.form[feature]])[0]
            else:
                return render_template('liver_result.html', prediction=f"Error: Invalid input for {feature}")

        # Convert input into DataFrame
        input_df = pd.DataFrame([inputs])

        # Reorder columns to match training order
        input_df = input_df[liver_model.feature_names_in_]

       
        input_df = input_df.astype(float)

        
        prediction = liver_model.predict(input_df)

        return render_template('liver_result.html', prediction=f"Predicted Cirrhosis Stage: {prediction[0]}")

    except Exception as e:
        return render_template('liver_result.html', prediction=f"Error: {str(e)}")



'''-----------------------------------------------------------------------------------------'''
medicines_dict = pickle.load(open('MD/medicine_dict.pkl', 'rb'))
medicines = pd.DataFrame(medicines_dict)
similarity = pickle.load(open('MD/similarity.pkl', 'rb'))


def recommend_medicine(medicine_name):
    try:
        medicine_index = medicines[medicines['Drug_Name'] == medicine_name].index[0]
        distances = similarity[medicine_index]
        medicines_list = sorted(
            list(enumerate(distances)), reverse=True, key=lambda x: x[1]
        )[1:6]

        recommended_medicines = [
            medicines.iloc[i[0]].Drug_Name for i in medicines_list
        ]
        return recommended_medicines
    except IndexError:
        return ["Medicine not found."]
    except Exception as e:
        return [f"Error: {str(e)}"]


@app.route('/medicines', methods=['GET', 'POST'])
def medicine():
    recommendations = []
    if request.method == 'POST':
        selected_medicine = request.form['medicine']
        recommendations = recommend_medicine(selected_medicine)
    return render_template(
        'medicine.html',
        medicine_names=medicines['Drug_Name'].values,
        recommendations=recommendations
    )



# load databasedataset
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv('datasets/medications.csv')
diets = pd.read_csv("datasets/diets.csv")


# load model
svc = pickle.load(open('MD/svc.pkl','rb'))


def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis] ['workout']


    return desc,pre,med,die,wrkout

symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]



@app.route('/recommendations', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        
        if not symptoms or symptoms.strip().lower() == "symptoms":
            message = "Please enter valid symptoms (comma-separated)."
            return render_template('recommendation.html', message=message)

        # Clean and split symptoms
        user_symptoms = [s.strip().lower() for s in symptoms.split(',')]
        user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]

        # Check for invalid symptoms
        invalid_symptoms = [sym for sym in user_symptoms if sym not in symptoms_dict]
        if invalid_symptoms:
            message = f"The following symptoms are not recognized: {', '.join(invalid_symptoms)}"
            return render_template('recommendation.html', message=message)

        predicted_disease = get_predicted_value(user_symptoms)
        dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)

        my_precautions = [i for i in precautions[0]]

        return render_template('recommendation.html',
                               predicted_disease=predicted_disease,
                               dis_des=dis_des,
                               my_precautions=my_precautions,
                               medications=medications,
                               my_diet=rec_diet,
                               workout=workout,
                               user_symptoms=user_symptoms)  # <-- Add this line

    return render_template('recommendation.html')

db.init_app(app)  
if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Ensures tables are created
    app.run(debug=True, threaded=False)