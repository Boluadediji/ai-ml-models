import os
import logging
import joblib
import numpy as np
import json
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)

def init():

    # Initialize the model when the container starts

    global model, vectorizer

    try:
        # AZUREML_MODEL_DIR is provided by Azure
        model_dir = os.getenv('AZUREML_MODEL_DIR') or "."
        logging.info(f" model directory: {model_dir}")

        if model_dir is None or not os.path.exists(model_dir):
            model_dir = '.'

        # List files to see what is available
        if model_dir and os.path.exists(model_dir):
            logging.info(f"Files in model dir: {os.listdir(model_dir)}")

        # Load the model
        model_path = os.path.join(model_dir, "bank_model.pkl")

        if not os.path.exists(model_path):
            # fallback: try current directory
            model_path = "bank_model.pkl"
            logging.info("Trying current directory for model file.")

        logging.info(f"Loading model from: {model_path}")

        # Load model using joblib
        with open(model_path, "rb") as f:
            # Your model is stored as (vectorizer, model) tuple
            vectorizer, model = joblib.load(f)

        logging.info("Model loaded successfully.")
        logging.info(f"Vectorizer type: {type(vectorizer)}")
        logging.info(f"Model type: {type(model)}")

        # Test prediction with dummy data
        test_text = "check my balance"
        X_test = vectorizer.transform([test_text])
        test_pred = model.predict(X_test)[0]
        logging.info(f"Test prediction successful: {test_pred}")


    except Exception as e:
        logging.error(f"Model loading failed: {str(e)}")
        logging.error(traceback.format_exc())
        raise



def run(raw_data):

    # Handle prediction requests

    try:
        logging.info(f"Received prediction requests")

        # Parse input data
        data = json.loads(raw_data)

        # Get text from request
        text = data.get("text", "")

        if not text:
            #Alternative: Check if data is directly the text
            if isinstance(data, str):
                text = data
            else:
                return {"error": "No text provided in 'text' filed"}
            
        logging.info(f"Processing text: {text}")


        # Transform and predict
        X = vectorizer.transform([text])
        prediction = model.predict(X)[0]

        # Get probabilities if available
        probabilities = []
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X).tolist()[0]

        # Return results
        result = {
            "prediction": str(prediction),
            "probabilities": probabilities
        }

        logging.info(f"Prediction: {result}")
        return result

    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        logging.error(f"{error_msg}")
        return {"error": error_msg}
    

if __name__ == "__main__":
    print("Testing score.py locally...")
    try:
        init()
        
        # Test cases
        test_cases = [
            '{"text": "I want to check my balance"}',
            '{"text": "Transfer money to my savings"}',
            '{"text": "What is my account balance?"}'
        ]
        
        for test_data in test_cases:
            print(f"\nTesting: {test_data}")
            result = run(test_data)
            print(f"Result: {result}")
            
    except Exception as e:
        print(f"Local test failed: {e}")
        print(traceback.format_exc())