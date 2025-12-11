# api.py
import os
import json
import asyncio
import traceback
from typing import Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Body, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import base64

from text_handler import handle_text_message
from call import handle_live_call, process_user_utterance
from config import get_azure_client
from intent_classifier import classify_intent
from ragengine import retrieve_relevant_doc
from call import speech_config, TTSPushCallback

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(title="Alex Banking AI", version="1.0")

# -----------------------
# Session storage (simple in-memory, swap for Redis in prod)
# -----------------------
SESSIONS: Dict[str, Dict[str, Any]] = {}
MAX_HISTORY = 8

def get_session(session_id: str) -> Dict[str, Any]:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {"history": [], "intent": None}
    return SESSIONS[session_id]

def persist_session(session_id: str, session: Dict[str, Any]):
    session["history"] = session.get("history", [])[-MAX_HISTORY:]
    SESSIONS[session_id] = session

# -----------------------
# Helpers
# -----------------------
def b64_audio(audio_bytes: bytes) -> str:
    return base64.b64encode(audio_bytes).decode("utf-8")

def decode_b64_audio(b64_str: str) -> bytes:
    return base64.b64decode(b64_str.encode("utf-8"))

async def maybe_await(func, *args, **kwargs):
    if asyncio.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    else:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

# -----------------------
# Request models
# -----------------------
class TextChatRequest(BaseModel):
    session_id: str
    message: str

class TextChatResponse(BaseModel):
    reply: str
    intent: Optional[str]
    escalation: Optional[bool] = False

class VoiceTurnRequest(BaseModel):
    session_id: Optional[str] = None

class LLMRequest(BaseModel):
    text: str
    intent: Optional[str] = None

class TTSRequest(BaseModel):
    text: str

# -----------------------
# Azure client
# -----------------------
client = get_azure_client()

# -----------------------
# TEXT CHAT
# -----------------------
@app.post("/chat/text", response_model=TextChatResponse)
async def chat_text(payload: TextChatRequest):
    try:
        session = get_session(payload.session_id)
        reply, escalation, updated_session = await maybe_await(
            handle_text_message, payload.message, session
        )
        updated_session["intent"] = updated_session.get("session_intent")
        persist_session(payload.session_id, updated_session)
        return TextChatResponse(reply=reply, intent=updated_session["intent"], escalation=escalation)
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Text chat failed")

# -----------------------
# FULL VOICE TURN (ASR → Intent → RAG → LLM → TTS)
# -----------------------
@app.post("/voice/turn")
async def voice_turn(session_id: Optional[str] = Body(None), audio: UploadFile = File(...)):
    try:
        audio_bytes = await audio.read()
        # process_user_utterance expects client and a send_audio callback
        collected_audio = bytearray()
        def send_audio_chunk(chunk: bytes):
            collected_audio.extend(chunk)

        # call live handler directly for one turn
        await process_user_utterance(
            user_text=(await maybe_await(lambda b: b.decode("utf-8"), audio_bytes)),  # decode placeholder
            client=client,
            send_audio_chunk=send_audio_chunk
        )

        # persist session
        if session_id:
            session = get_session(session_id)
            session["history"].append({"user": "audio_input", "bot": "audio_response"})
            persist_session(session_id, session)

        return {"audio_b64": b64_audio(collected_audio), "content_type": "audio/wav"}
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Voice turn failed")

# -----------------------
# LIVE CALL WS
# -----------------------
@app.websocket("/ws/livecall")
async def ws_livecall(ws: WebSocket):
    await ws.accept()
    async def get_audio_chunk():
        try:
            while True:
                data = await ws.receive_bytes()
                yield data
        except WebSocketDisconnect:
            return
    async def send_audio_chunk(audio_bytes: bytes):
        await ws.send_bytes(audio_bytes)
    try:
        await handle_live_call(get_audio_chunk, send_audio_chunk, client)
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception:
        traceback.print_exc()

# -----------------------
# HEALTH
# -----------------------
@app.get("/health")
def health():
    return {"status": "ok"}
