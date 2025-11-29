import os, json, requests
import pickle
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_classic.agents import AgentExecutor, Tool
from langchain_classic.prompts import PromptTemplate
import azure.cognitiveservices.speech as speech
from langchain_classic.agents.mrkl.base import ZeroShotAgent

load_dotenv()

def classify_intent(text:str):
    model_api_endpoint = os.getenv("model_endpoint")
    model_api_key = os.getenv("model_api_key")

    response = requests.post(
        url=model_api_endpoint,
        json={"text": text},
        headers={"Content-Type":"application/json", "Authorization": f"Bearer {model_api_key}"}
    )

    data = dict(response.json())
    print(data['prediction'])
    return data['prediction']


def azure_llm():
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("azure_resource_endpoint"),
        api_key=os.getenv("azure_resource_key"),
        api_version="2024-12-01-preview",
        max_tokens=4096,
        temperature=0,
        azure_deployment="gpt-4o-mini"
    )

    return llm

def extract_account_id(text:str):
    import re
    match = re.search(r'\b(\d{3})\b', text)
    return match.group(1) if match else None

def check_balance(account_id:str):
    global accounts
    acct = accounts.get(extract_account_id(account_id))
    if not acct:
        return {"error":"Account not found."}
    return {"account_id":account_id, "balance":acct["balance"]}
    
def report_card_issues(account_id:str):
    # Simulate blocking card
    return {"status":"blocked", "account_id":account_id, "next_step":"Collect new card in 48 hours"}

def unsupported(text:str=""):
    return {"status":"unsupported"}