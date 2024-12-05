# QueryFixer

## Frontend:

The frontend is a simple web application that asks for a search query. When submitted, the frontend will display whether the query is well-formed or not and if it is not, it will offer a well-formed version of the query.

### To start the frontend:

Run the following commands:

        cd Frontend
        python -m http.server 8000

Alternatively, you can follow these steps:

1. Open VS Code and ensure you have the Live Server extension installed.

2. Right click on the Frontend/index.html file and select the live server option.


## Backend API:

The backend api endpoint will accept a query, pass the query through a pre-trained binary classification model, and respond with whether that query is well-formed or not well-formed. If the query is not well-formed, it will provided a suggested query that is well-formed.

### To start the classifier Flask API:

1. Install neccessary packages.
        
        pip install -r ./Backend/requirements.txt

2. Create a .env file in ./Backend that follows the format in the /Backend/.env.example file. This step requires an open AI API key. NOTE: If no API key is supplied, the backend will still run, but will not supply any query suggestions for malformed queries.

3. Ensure you are in the main folder and run the following command to start the development server:

       python ./Backend/query_fixer_backend.py

### Endpoint details:

#### URL: 
    POST http://127.0.0.1:5000/evaluate-query

#### Request Body: 
    {
        "query": "sample query"
    }

#### Example curl command:

    curl -X POST -H "Content-Type: application/json" -d '{"query": "what cat?" }' http://127.0.0.1:5000/evaluate-query

#### Example Response Body:
    {
        "is_well_formed": False, 
        "suggested_query": "What is a cat?"
    }



