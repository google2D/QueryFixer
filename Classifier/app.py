from flask import Flask, request, jsonify
from Model import Model
from Encoder import Encoder

# Initialize Flask app
app = Flask(__name__)

# Load the model and initialize the bert encoder once when the server starts
model = Model("model.pth")
encoder = Encoder()

@app.route('/classify', methods=['POST'])
def classify():
    try:
        # Get the JSON data from the request
        data = request.get_json()
        
        # Extract the query parameter 
        query = data.get('query', '')  
        
        # Check if the query parameter exists
        if not query:
            return jsonify({"error": "No input text provided"}), 400
        
        # Encode the query
        embedded_query_tensor = encoder.bert_encode_sequence(query)

        if model.is_well_formed(embedded_query_tensor):
            result = {"is_well_formed": True, "message": "The sequence is well-formed."}
        else:
            result = {"is_well_formed": False, "message": "The sequence is not well-formed."}
        
        # Return the result as a JSON response
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
