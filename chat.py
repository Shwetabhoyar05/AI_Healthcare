import google.generativeai as genai

def get_health_recommendation(user_data, prediction_result):
    prompt = f"""
You are a virtual health assistant. A patient has submitted the following medical information:

Age: {user_data['age']}
Sex: {user_data['sex']}
Chest Pain Type: {user_data['chest pain type']}
Resting Blood Pressure (systolic): {user_data['resting bp s']}
Cholesterol: {user_data['cholesterol']}
Fasting Blood Sugar: {user_data['fasting blood sugar']}
Resting ECG: {user_data['resting ecg']}
Max Heart Rate: {user_data['max heart rate']}
Exercise-Induced Angina: {user_data['exercise angina']}
Oldpeak (ST depression): {user_data['oldpeak']}
ST Slope: {user_data['ST slope']}

The model predicts: {prediction_result}

Now, provide a clear, structured, and point-wise personalized lifestyle recommendation for the user that includes:

1. **Diet Advice**
2. **Physical Activity**
3. **Stress Management**
4. **Warning Signs to Monitor**

Each section should have bullet points (•) and avoid paragraphs. Be friendly and concise. Maximum 5 bullet points per section.
"""

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()





'''-------------------------------------------------------------------------------------'''
import  requests
genai.configure(api_key="API KEY")

# Fetch dataset
def fetch_dataset():
    url = "https://datasets-server.huggingface.co/rows?dataset=jovan-antony%2Fhealthcare_chatbot_dataset&config=default&split=train&offset=0&length=100"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()["rows"]
        return {
            row["row"]["Symptom"].lower(): {
                "Disease": row["row"]["Disease"],
                "Treatment": row["row"]["Treatment"],
            }
            for row in data
        }
    else:
        print("⚠️ Failed to fetch dataset. Using default values.")
        return {
            "fever": {"Disease": "flu", "Treatment": "rest, hydration, antiviral medications"},
            "rash": {"Disease": "allergy", "Treatment": "avoid allergens, antihistamines, consult doctor"},
            "vomiting": {"Disease": "food poisoning", "Treatment": "rest, hydration, avoid solid food, consult doctor if severe"},
        }

symptom_disease_treatment = fetch_dataset()

def ask_gemini(query):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
You are an AI health assistant. Answer the following query clearly using:
- Bullet points (• or -) where needed
- No unnecessary asterisks or Markdown formatting

Query: {query}
    """
    response = model.generate_content(prompt)
    if response.text:
        formatted = response.text.strip().replace("\n\n", "<br><br>").replace("\n", "<br>").replace("*", "")
        return formatted
    return "Sorry, I couldn't generate a proper response. 😥"


def healthcare_chatbot(user_input):
    user_input = user_input.lower().strip()

    # Small talk / general question handling
    small_talk_responses = {
        "hi": "👋 Hello! How can I assist you today?",
        "hello": "👋 Hello! What symptoms are bothering you?",
        "hey": "Hey there! 😊 Tell me how you're feeling.",
        "how are you": "I'm just a bunch of code, but I'm here to help you stay healthy! 💪",
        "good morning": "☀️ Good morning! How can I help you today?",
        "good evening": "🌆 Good evening! Tell me your symptoms.",
        "thanks": "You're welcome! 😊 Stay healthy!",
        "thank you": "No problem! Let me know if you need anything else.",
    }

    for phrase, response in small_talk_responses.items():
        if phrase in user_input:
            return response

    # Symptom-based response
    if user_input in symptom_disease_treatment:
        disease = symptom_disease_treatment[user_input]["Disease"]
        treatment = symptom_disease_treatment[user_input]["Treatment"]
        gemini_advice = ask_gemini(f"What are additional home remedies for {disease}?")

        response = f"""
        🤒 <strong>Symptom:</strong> {user_input.capitalize()}<br>
        🦠 <strong>Possible Disease:</strong> {disease.capitalize()}<br>
        💊 <strong>Suggested Treatment:</strong> {treatment}<br><br>
        🏥 <strong>Gemini Advice:</strong> {gemini_advice}
        """
    else:
        
        response = ask_gemini(f"What could be the possible causes and treatments for {user_input}?")

    return response
