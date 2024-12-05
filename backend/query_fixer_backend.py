from flask import Flask, request, jsonify
from Classifier.Model import Model
from Classifier.Encoder import Encoder
from dotenv import load_dotenv
from flask_cors import CORS

import os
import openai  

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the .env file and set open AI API key
load_dotenv()
openai.api_key = os.getenv('OPEN_AI_API_KEY')

# Initialize model and encoder classes
model = Model()
encoder = Encoder()

@app.route("/evaluate-query", methods=["POST"])
def evaluate_query():
    try:
        # Get the JSON data from the request
        data = request.get_json()
        
        # Extract the query parameter 
        query = data.get('query', '')  
        
        # Check if the query parameter exists
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        # Encode the query
        embedded_query_tensor = encoder.bert_encode_sequence(query)

        # Pass query through classifier model to check if it is well-formed
        is_well_formed = True if model.is_well_formed(embedded_query_tensor) else False
        
        # Generate a suggestion if not well-formed
        suggested_query = ""
        if not is_well_formed and openai.api_key:
            response = openai.Completion.create(
                model="text-davinci-003",  # Replace with the relevant OpenAI model
                prompt=f"Fix this search query to be well-formed: {query}",
                max_tokens=50
            )
            suggested_query = response.choices[0].text.strip()

     
        return jsonify({
            "is_well_formed": is_well_formed,
            "suggested_query": suggested_query
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
