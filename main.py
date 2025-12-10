from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ragengine import retrieve_relevant_doc
from config import get_azure_client, CHAT_MODEL
import uvicorn
import os

app = FastAPI()
client = get_azure_client()

# Data model for the request (Used by both Chat and Voice)
class AIRequest(BaseModel):
    intent: str = "general" # Default to general if unknown
    message: str            # The user's question (typed or spoken)

@app.get("/health")
def health():
    return {"status": "healthy", "service": "AI-Brain"}

@app.post("/generate")
def generate_response(request: AIRequest):
    """
    This SINGLE endpoint handles both:
    1. Text Chat (Directly from mobile)
    2. Voice Chat (Transcribed text from Backend)
    """
    print(f"Processing Request: {request.message} | Intent: {request.intent}")
    
    try:
        # 1. Retrieve Knowledge (RAG) - Reuses your existing ragengine.py
        #
        doc_content, score = retrieve_relevant_doc(request.intent, request.message)
        
        # 2. Build Prompt
        system_prompt = f"""
        You are Alex, Wema Bank's support assistant.
        Use this bank policy to answer:
        {doc_content if doc_content else "No specific policy found."}
        
        Keep answers short, professional, and empathetic.
        """

        # 3. Call GPT (The Brain)
        response = client.chat.completions.create(
            model=CHAT_MODEL, # Uses "aicso" from config.py
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.message}
            ],
            temperature=0,
            max_tokens=150 # Keep it short for voice!
        )
        
        reply_text = response.choices[0].message.content

        return {
            "response": reply_text,
            "source_doc": "policy_found" if doc_content else "general_knowledge",
            "confidence": score
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)