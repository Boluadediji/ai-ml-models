from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from ragengine import retrieve_relevant_doc
from config import get_azure_client, CHAT_MODEL
import uvicorn
import os

app = FastAPI()
client = get_azure_client()

# --- 1. Define Models matching C# DTOs ---
class AIMessage(BaseModel):
    role: str
    content: str

class AIRequest(BaseModel):
    user_text: str                  # Matches C# 'user_text'
    session_intent: str = "general" # Matches C# 'session_intent'
    conversation_history: List[AIMessage] = []

@app.get("/health")
def health():
    return {"status": "healthy", "service": "AI-Brain"}

@app.post("/generate")
def generate_response(request: AIRequest):
    print(f"Processing: {request.user_text} | Intent: {request.session_intent}")
    
    try:
        # 1. Retrieve Knowledge
        # Note: We use request.user_text now (not request.message)
        doc_content, score = retrieve_relevant_doc(request.session_intent, request.user_text)
        
        # 2. Build Prompt
        system_prompt = f"""
        You are Alex, Wema Bank's support assistant.
        Use this bank policy to answer if relevant:
        {doc_content if doc_content else "No specific policy found."}
        
        Keep answers short, professional, and empathetic.
        """

        # 3. Call GPT
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add history from C# (optional, but good for context)
        for msg in request.conversation_history:
            messages.append({"role": msg.role, "content": msg.content})
            
        messages.append({"role": "user", "content": request.user_text})

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=150
        )
        
        reply_text = response.choices[0].message.content

        # 4. Return JSON matching C# AIResponse
        return {
            "reply": reply_text,                         # Matches C# 'reply'
            "session_intent": request.session_intent,    # Matches C# 'session_intent'
            "conversation_history": request.conversation_history # Matches C#
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)