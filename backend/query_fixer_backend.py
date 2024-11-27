from flask import Flask, request, jsonify
from your_ml_model import classify_query  # Your ML model logic
import openai  # OpenAI library for GPT integration

# Initialize Flask app
app = Flask(__name__)

# OpenAI API Key
openai.api_key = "your-openai-api-key"

@app.route("/evaluate-query", methods=["POST"])
def evaluate_query():
    data = request.json
    user_query = data.get("query", "")

    if not user_query:
        return jsonify({"error": "Query cannot be empty"}), 400

    # ML Model: Check if the query is well-formed
    is_well_formed = classify_query(user_query)  # Replace with your model logic

    # Generate a suggestion if not well-formed
    suggested_query = None
    if not is_well_formed:
        response = openai.Completion.create(
            model="text-davinci-003",  # Replace with the relevant OpenAI model
            prompt=f"Fix this search query to be well-formed: {user_query}",
            max_tokens=50
        )
        suggested_query = response.choices[0].text.strip()

    return jsonify({
        "isWellFormed": is_well_formed,
        "suggestedQuery": suggested_query
    })

if __name__ == "__main__":
    app.run(debug=True)
