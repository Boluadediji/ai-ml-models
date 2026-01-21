# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import List, Optional
# from ragengine import retrieve_relevant_doc
# from config import get_azure_client, CHAT_MODEL
# import uvicorn
# import os

# app = FastAPI()
# client = get_azure_client()

# # --- 1. Define Models matching C# DTOs ---
# class AIMessage(BaseModel):
#     role: str
#     content: str

# class AIRequest(BaseModel):
#     user_text: str                  # Matches C# 'user_text'
#     session_intent: str = "general" # Matches C# 'session_intent'
#     conversation_history: List[AIMessage] = []

# @app.get("/health")
# def health():
#     return {"status": "healthy", "service": "AI-Brain"}

# @app.post("/generate")
# def generate_response(request: AIRequest):
#     print(f"Processing: {request.user_text} | Intent: {request.session_intent}")
    
#     try:
#         # 1. Retrieve Knowledge
#         # Note: We use request.user_text now (not request.message)
#         doc_content, score = retrieve_relevant_doc(request.session_intent, request.user_text)
        
#         # 2. Build Prompt
#         system_prompt = f"""
#         You are Alex, Wema Bank's support assistant.
#         Use this bank policy to answer if relevant:
#         {doc_content if doc_content else "No specific policy found."}
        
#         Keep answers short, professional, and empathetic.
#         """

#         # 3. Call GPT
#         messages = [{"role": "system", "content": system_prompt}]
        
#         # Add history from C# (optional, but good for context)
#         for msg in request.conversation_history:
#             messages.append({"role": msg.role, "content": msg.content})
            
#         messages.append({"role": "user", "content": request.user_text})

#         response = client.chat.completions.create(
#             model=CHAT_MODEL,
#             messages=messages,
#             temperature=0,
#             max_tokens=150
#         )
        
#         reply_text = response.choices[0].message.content

#         # 4. Return JSON matching C# AIResponse
#         return {
#             "reply": reply_text,                         # Matches C# 'reply'
#             "session_intent": request.session_intent,    # Matches C# 'session_intent'
#             "conversation_history": request.conversation_history # Matches C#
#         }

#     except Exception as e:
#         print(f"Error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)


from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List
import uvicorn
import base64

# Import his logic (assuming these files exist in his update)
# from chat_service import process_message
# from voice_service import handle_voice_logic_if_needed 

# Import your existing logic
from ragengine import retrieve_relevant_doc
from config import get_azure_client, CHAT_MODEL

app = FastAPI()
client = get_azure_client()

# --- DTOs matching C# ---
class AIRequest(BaseModel):
    user_text: str
    session_intent: str = "general"
    conversation_history: List[dict] = []

class TTSRequest(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "healthy"}

# --- Endpoint 1: The Brain (Matches C# AIServiceClient) ---
@app.post("/generate")
def chat(request: AIRequest):
    print(f"Thinking about: {request.user_text}")
    try:
        # 1. RAG Retrieval
        doc_content, score = retrieve_relevant_doc(request.session_intent, request.user_text)
        
        # 2. GPT Generation
        system_prompt = f"You are Alex, Wema Bank support. Policy: {doc_content}"
        
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.user_text}
            ],
            max_tokens=150
        )
        
        return {
            "reply": response.choices[0].message.content,
            "session_intent": request.session_intent,
            "conversation_history": request.conversation_history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Endpoint 2: The Ears (Matches C# RealASRService) ---
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    print("Transcribing audio...")
    # In reality, you'd use Whisper or Azure Speech here
    # return {"transcript": transcribe_audio(await file.read())}
    
    return {"transcript": "I want to transfer money"} # Mock until he gives you voice.py

# --- Endpoint 3: The Mouth (Matches C# RealTTSService) ---
@app.post("/synthesize")
def synthesize(request: TTSRequest):
    print(f"Speaking: {request.text}")
    # In reality, you'd generate audio bytes here
    
    dummy_audio = b"fake_audio_content"
    return {"audio_base64": base64.b64encode(dummy_audio).decode("utf-8")}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
