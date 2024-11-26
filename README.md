# QueryFixer


## Classifer:

### To start classifier Flask API:

1. Install neccessary packages.

2. run python ./Classifier/app.py

### Endpoint details:

#### URL: 
POST http://127.0.0.1:5000/classify

#### Request Body: 
{
  "query": "sample query"
}

#### Example curl command:

curl -X POST -H "Content-Type: application/json" -d '{"query": "what is a cat?" }' http://127.0.0.1:5000/classify

#### Example Response Body:
{
    "is_well_formed": True, 
    "message": "The sequence is well-formed."
}



