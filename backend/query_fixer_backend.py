from flask import Flask, request, jsonify
from flask_cors import CORS
import openai

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Set your OpenAI API key
openai.api_key = '':

@app.route('/fix-query', methods=['POST'])
def fix_query():
    # Get the query from the request payload
    user_query = request.json.get('query')

    if not user_query:
        return jsonify({'error': 'No query provided'}), 400

    # Use OpenAI API to generate a fixed query
    try:
        response = openai.Completion.create(
            model="gpt-4o-mini",  # Or your preferred model
            prompt=f"Fix this search query: {user_query}",
            max_tokens=50
        )

        # Get the fixed query from OpenAI response
        fixed_query = response.choices[0].text.strip()

        return jsonify({'original': user_query, 'fixed': fixed_query})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
