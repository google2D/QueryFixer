import joblib

# Load the trained ML model
model = joblib.load("path_to_your_trained_model.pkl")
vectorizer = joblib.load("path_to_your_vectorizer.pkl")

def classify_query(query):
    transformed_query = vectorizer.transform([query])
    prediction = model.predict(transformed_query)
    return prediction[0] == 1  # Assuming 1 represents "well-formed"
