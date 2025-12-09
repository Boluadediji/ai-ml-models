# config.py

import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# -----------------------------
# Environment Variables
# -----------------------------
AZURE_ENDPOINT = os.getenv("azure_resource_endpoint")
AZURE_API_KEY = os.getenv("azure_resource_key")
AZURE_API_VERSION = "2024-12-01-preview"

# -----------------------------
# Model Settings
# -----------------------------
CHAT_MODEL = "gpt-4o-mini"

# -----------------------------
# Document Paths
# -----------------------------
TRANSFER_POLICY_PATH = "documents/transfer_policy.txt"
CARD_POLICY_PATH = "documents/card_policy.txt"

# -----------------------------
# Lazy Client Loader (Singleton)
# -----------------------------
_client = None


def get_azure_client():
    """
    Returns a single AzureOpenAI client instance.
    Creates it only once (lazy initialization).
    """
    global _client

    if _client is None:
        _client = AzureOpenAI(
            api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
        )
    return _client
