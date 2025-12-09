from config import *
import os
import requests

def classify_intent(text: str):
    model_api_endpoint = os.getenv("model_endpoint")
    model_api_key = os.getenv("model_api_key")
    response = requests.post(url=model_api_endpoint, json={"text": text}, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {model_api_key}"
    })

    data = dict(response.json())
    # print(data["prediction"])

    # if data["prediction"] == "transaction":
    #     data["prediction"] = "transfer_issue"

    if (max(data["probabilities"]) < 0.6):
        data["prediction"] = "unsupported_request"

    return data["prediction"]