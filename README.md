# AI_Healthcare Overview

AI_Healthcare is a web application that uses machine learning to predict risks for heart disease, lung cancer survival, and liver cirrhosis stages based on user data input. It features a symptom analyzer that suggests possible diseases along with recommendations for medications, precautions, and diet, a medicine recommender for a list of 5 alternative options, and an AI chatbot for medical queries. Built with Flask, Python, and SQLite, the web app provides personalized health insights through trained Random Forest and Decision Tree models, while ensuring secure user access via login authentication. AI_Healthcare combines medical AI with user-friendly interfaces to support better health decisions.
# _________________________________________________________

# Features

* **User-Friendly Interface**: Clean, responsive UI ensures easy navigation across devices for seamless health predictions and recommendations.
* **Authentication & Security**: Secure login/signup with encrypted data handling protects user privacy and prevents unauthorized access.
* **Heart Disease Prediction**: Analyzes age, BP, and cholesterol to predict heart disease risk and suggests personalized health routines.
* **Lung Cancer Survival Prediction**: Estimates survival chances based on smoking status, cancer stage, and overall health metrics.
* **Liver Cirrhosis Stage Prediction**: Predicts cirrhosis progression using liver function tests to guide treatment decisions.
* **AI Medical Chatbot**: Gemini API-powered assistant answers medical queries, explains symptoms, and offers health advice.
* **Symptom-Based Disease Recommendation**: Input symptoms (e.g., skin rashes) to get possible diagnoses with medication, diet, and precaution tips.
* **Medicine Recommendation**: Suggests 5 alternative medicines for any searched drug, aiding treatment flexibility.
# _________________________________________________________

# Technology Stack
* **Frontend**: HTML, CSS, JavaScript(Basic), Bootstrap.

* **Backend**: Python,Flask.

* **Database**:SQLite,SQLAlchemy (ORM)

* **Machine Learning Model**: scikit-learn(	Random Forest, Decision Trees,support vector classifier),Gemini API(gemini-1.5-flash),Pickle (for model serialization)

* **Deployment**:Flask (for local hosting)

* **Version Control**:GitHub
# _________________________________________________________


# Installation

### 1. Clone the Repository

```bash
  git clone https://github.com/Shwetabhoyar05/AI_Healthcare.git
  cd AI_Healthcare
```
### 2. Set Up Virtual Environment

```bash
  python -m venv AIenv
  AIenv\Scripts\activate
```
### 3. Install Dependencies

```bash
  pip install -r requirements.txt
```

### 4. Initialize Database

```bash
  python Models.py
```

### 5. Run the Application

```bash
 python Main.py
```

### 6. Access the App

```bash
 🌐 Open your browser and visit: http://localhost:5000
```
# _________________________________________________________
    
# 📁 Project File Structure

```
HealthAI/
├── Datasets/                   # Training data and reference files
│   ├── Symptom-severity.csv
│   ├── Training.csv
│   ├── dataset.csv
│   ├── description.csv
│   ├── diets.csv
│   ├── medications.csv
│   ├── precautions_df.csv
│   ├── symtoms_df.csv
│   └── workout_df.csv
├── MD/                         # Machine learning assets
│   ├── Pickle-files.rar
│   ├── medicine_dict.pkl
│   ├── similarity.pkl
│   └── svc.pkl
├── Models.py                   # Database models and setup
├── Main.py                     # Main application entry point
├── chat.py                     # Chatbot backend functions
├── models/                     # Pretrained ML models
│   ├── Liver_Cirrhosis_Stage_Detection_DT.pkl
│   ├── heart_disease_model.pkl
│   ├── heart_scaler.pkl
│   ├── liver_label_encoders.pkl
│   ├── lung_cancer_model.pkl
│   ├── lung_label_encoders.pkl
│   └── lung_scaler.pkl
├── static/
│   └── images/                 # Application images
├── templates/                  # HTML templates
│   ├── chatbot.html            # Chatbot interface
│   ├── heart_input.html        # Heart disease input form
│   ├── heart_result.html       # Heart disease results
│   ├── home.html               # Main page
│   ├── index.html              # Landing page
│   ├── liver_input.html        # Liver cirrhosis input form
│   ├── liver_result.html       # Liver results
│   ├── login.html              # Login page
│   ├── lungs_input.html        # Lung cancer input form
│   ├── lungs_result.html       # Lung cancer results
│   ├── medicine.html           # Medicine recommender
│   ├── recommendation.html     # Symptom analyzer
│   └── signup.html             # Registration page
├── requirements.txt            # Python dependencies
└── .env                        # Environment configuration
```
# _________________________________________________________

# 🤝 Contributing

Contributions are welcome! Feel free to fork the repository and submit a pull request to improve the project.
# _________________________________________________________

# Contact

 For any queries or suggestions, feel free to reach out:

GitHub: https://github.com/Shwetabhoyar05



