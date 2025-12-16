# # api.py
# import os
# import json
# import asyncio
# import traceback
# from typing import Optional, Dict, Any
# from fastapi import FastAPI, UploadFile, File, Body, HTTPException, WebSocket, WebSocketDisconnect
# from pydantic import BaseModel
# import base64

# from text_handler import handle_text_message
# from call import handle_live_call, process_user_utterance
# from config import get_azure_client
# from intent_classifier import classify_intent
# from ragengine import retrieve_relevant_doc
# from call import speech_config, TTSPushCallback

# # -----------------------
# # FastAPI app
# # -----------------------
# app = FastAPI(title="Alex Banking AI", version="1.0")

# # -----------------------
# # Session storage (simple in-memory, swap for Redis in prod)
# # -----------------------
# SESSIONS: Dict[str, Dict[str, Any]] = {}
# MAX_HISTORY = 8

# def get_session(session_id: str) -> Dict[str, Any]:
#     if session_id not in SESSIONS:
#         SESSIONS[session_id] = {"history": [], "intent": None}
#     return SESSIONS[session_id]

# def persist_session(session_id: str, session: Dict[str, Any]):
#     session["history"] = session.get("history", [])[-MAX_HISTORY:]
#     SESSIONS[session_id] = session

# # -----------------------
# # Helpers
# # -----------------------
# def b64_audio(audio_bytes: bytes) -> str:
#     return base64.b64encode(audio_bytes).decode("utf-8")

# def decode_b64_audio(b64_str: str) -> bytes:
#     return base64.b64decode(b64_str.encode("utf-8"))

# async def maybe_await(func, *args, **kwargs):
#     if asyncio.iscoroutinefunction(func):
#         return await func(*args, **kwargs)
#     else:
#         loop = asyncio.get_running_loop()
#         return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

# # -----------------------
# # Request models
# # -----------------------
# class TextChatRequest(BaseModel):
#     session_id: str
#     message: str

# class TextChatResponse(BaseModel):
#     reply: str
#     intent: Optional[str]
#     escalation: Optional[bool] = False

# class VoiceTurnRequest(BaseModel):
#     session_id: Optional[str] = None

# class LLMRequest(BaseModel):
#     text: str
#     intent: Optional[str] = None

# class TTSRequest(BaseModel):
#     text: str

# # -----------------------
# # Azure client
# # -----------------------
# client = get_azure_client()

# # -----------------------
# # TEXT CHAT
# # -----------------------
# @app.post("/chat/text", response_model=TextChatResponse)
# async def chat_text(payload: TextChatRequest):
#     try:
#         session = get_session(payload.session_id)
#         reply, escalation, updated_session = await maybe_await(
#             handle_text_message, payload.message, session
#         )
#         updated_session["intent"] = updated_session.get("session_intent")
#         persist_session(payload.session_id, updated_session)
#         return TextChatResponse(reply=reply, intent=updated_session["intent"], escalation=escalation)
#     except Exception:
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail="Text chat failed")

# # -----------------------
# # FULL VOICE TURN (ASR → Intent → RAG → LLM → TTS)
# # -----------------------
# @app.post("/voice/turn")
# async def voice_turn(session_id: Optional[str] = Body(None), audio: UploadFile = File(...)):
#     try:
#         audio_bytes = await audio.read()
#         # process_user_utterance expects client and a send_audio callback
#         collected_audio = bytearray()
#         def send_audio_chunk(chunk: bytes):
#             collected_audio.extend(chunk)

#         # call live handler directly for one turn
#         await process_user_utterance(
#             user_text=(await maybe_await(lambda b: b.decode("utf-8"), audio_bytes)),  # decode placeholder
#             client=client,
#             send_audio_chunk=send_audio_chunk
#         )

#         # persist session
#         if session_id:
#             session = get_session(session_id)
#             session["history"].append({"user": "audio_input", "bot": "audio_response"})
#             persist_session(session_id, session)

#         return {"audio_b64": b64_audio(collected_audio), "content_type": "audio/wav"}
#     except Exception:
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail="Voice turn failed")

# # -----------------------
# # LIVE CALL WS
# # -----------------------
# @app.websocket("/ws/livecall")
# async def ws_livecall(ws: WebSocket):
#     await ws.accept()
#     async def get_audio_chunk():
#         try:
#             while True:
#                 data = await ws.receive_bytes()
#                 yield data
#         except WebSocketDisconnect:
#             return
#     async def send_audio_chunk(audio_bytes: bytes):
#         await ws.send_bytes(audio_bytes)
#     try:
#         await handle_live_call(get_audio_chunk, send_audio_chunk, client)
#     except WebSocketDisconnect:
#         print("Client disconnected")
#     except Exception:
#         traceback.print_exc()

# # -----------------------
# # HEALTH
# # -----------------------
# @app.get("/health")
# def health():
#     return {"status": "ok"}
# api.py

# api.py — FINAL, CLEAN, PRODUCTION VERSION (for .NET + gRPC + SignalR)

import base64
import traceback
from typing import Optional, List
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect, APIRouter
from pydantic import BaseModel
import uvicorn
from text_handler import handle_text_message
from voice import process_user_utterance, handle_live_call


app = FastAPI(title="Alex Banking AI", version="2.0")


# ---------------------------------------------------------
# 1. DTOs EXACTLY matching .NET
# ---------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    user_text: str
    conversation_history: List[ChatMessage] = []
    session_intent: Optional[str] = ""


class ChatResponse(BaseModel):
    reply: str
    conversation_history: List[ChatMessage]
    session_intent: str


class VoiceRequest(BaseModel):
    audio_b64: str
    content_type: str = "audio/wav"


class VoiceResponse(BaseModel):
    text: str
    intent: str
    reply: str
    audio_b64: Optional[str]
    content_type: str = "audio/wav"


# ---------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------

def decode_audio(b64_str: str) -> bytes:
    return base64.b64decode(b64_str)


def encode_audio(audio_bytes: bytes) -> str:
    return base64.b64encode(audio_bytes).decode()


# ---------------------------------------------------------
# 3. Chat Endpoint (SignalR-compatible)
# ---------------------------------------------------------

@app.post("/chat/text", response_model=ChatResponse)
async def chat_text(payload: ChatRequest):
    try:
        session_state = {
            "conversation_history": [
                {"role": msg.role, "content": msg.content}
                for msg in payload.conversation_history
            ],
            "session_intent": payload.session_intent,
        }

        reply, escalation, updated_state = handle_text_message(
            user_text=payload.user_text,
            session_state=session_state,
        )

        new_history = [
            ChatMessage(role=m["role"], content=m["content"])
            for m in updated_state["conversation_history"]
        ]

        return ChatResponse(
            reply=reply,
            conversation_history=new_history,
            session_intent=updated_state["session_intent"],
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# 4. VOICE ROUTES (File upload + Live Websocket)
# ---------------------------------------------------------

voice_router = APIRouter(prefix="/voice")


# ======================================================
# 1️⃣  VOICE-NOTE STYLE (WhatsApp style): /voice/process
# ======================================================
@voice_router.post("/process", response_model=VoiceResponse)
async def process_voice_file(file: UploadFile = File(...)):
    """
    Accepts audio (PCM/WAV/Opus/etc)
    Runs: STT → Intent → RAG → LLM → TTS
    Returns single-shot JSON for .NET mobile app.
    """
    audio_bytes = await file.read()

    result = await process_user_utterance(audio_bytes)

    return VoiceResponse(
        text=result["text"],
        intent=result["intent"],
        reply=result["reply"],
        audio_b64=encode_audio(result["audio"]),
        content_type="audio/wav",
    )


# ======================================================
# 2️⃣  TRUE LIVE CALL (Streaming) via WebSocket
# ======================================================
@voice_router.websocket("/live")
async def websocket_voice_call(ws: WebSocket):
    """
    True real-time call:
    - Client sends 20–60ms PCM/Opus chunks
    - Server streams TTS audio chunks back
    """

    await ws.accept()

    async def audio_in():
        """Yields incoming audio chunks."""
        while True:
            try:
                chunk = await ws.receive_bytes()
                yield chunk
            except WebSocketDisconnect:
                break
            except Exception:
                break

    async def audio_out(chunk: bytes):
        """Sends audio chunks back."""
        try:
            await ws.send_bytes(chunk)
        except:
            pass

    try:
        await handle_live_call(audio_in, audio_out)
    except WebSocketDisconnect:
        print("🔥 Client disconnected")
    except Exception as e:
        traceback.print_exc()
    finally:
        try:
            await ws.close()
        except:
            pass


# Register voice router
app.include_router(voice_router)


@app.get("/health")
def health():
    return {"status": "ok"}

# import os
# import json
# import base64
# import asyncio
# import traceback
# from typing import Optional, List, Dict, Any, Union
# from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
# from pydantic import BaseModel, Field

# # Import your logic handlers
# # Ensure these files exist in your container and handle the logic correctly
# from text_handler import handle_text_message
# from call import handle_live_call, process_user_utterance
# from config import get_azure_client

# app = FastAPI(title="Alex Banking AI", version="2.0")

# # -----------------------
# # Data Models (Matching .NET DTOs)
# # -----------------------

# # --- Chat Models ---
# class ChatMessage(BaseModel):
#     role: str
#     content: str

# class ChatRequest(BaseModel):
#     user_text: str
#     conversation_history: List[ChatMessage] = []
#     session_intent: Optional[str] = ""

# class ChatResponse(BaseModel):
#     reply: str
#     conversation_history: List[ChatMessage]
#     session_intent: Optional[str]

# # --- Voice Models ---
# class VoiceRequest(BaseModel):
#     audio_b64: str
#     content_type: str = "audio/wav"

# class VoiceResponse(BaseModel):
#     # Field aliases allow .NET capitalization (Text) while keeping Pythonic (text) if needed,
#     # but strictly returning snake_case usually works with .NET System.Text.Json default settings.
#     text: str
#     intent: str
#     reply: str
#     audio_b64: Optional[str] = None
#     content_type: str = "audio/wav"

# # -----------------------
# # Helpers
# # -----------------------
# def get_azure_services():
#     return get_azure_client()

# def decode_audio(b64_str: str) -> bytes:
#     return base64.b64decode(b64_str)

# def encode_audio(audio_bytes: bytes) -> str:
#     return base64.b64encode(audio_bytes).decode("utf-8")

# # -----------------------
# # CHAT ENDPOINT
# # -----------------------
# @app.post("/chat/text", response_model=ChatResponse)
# async def chat_text(payload: ChatRequest):
#     """
#     Handles text chat. State is reconstructed from payload.conversation_history.
#     """
#     try:
#         # 1. Reconstruct Session State from .NET History
#         # We convert the list of ChatMessage objects back to the dict structure your logic expects
#         history_for_logic = []
#         for msg in payload.conversation_history:
#             if msg.role.lower() in ["user"]:
#                 history_for_logic.append({"user": msg.content})
#             else:
#                 history_for_logic.append({"bot": msg.content})
        
#         # Create a temporary session object
#         temp_session = {
#             "history": history_for_logic,
#             "intent": payload.session_intent,
#             "session_intent": payload.session_intent # Handle legacy key if needed
#         }

#         # 2. Call Logic (Stateless)
#         # handle_text_message should take (text, session_dict) and return (reply, escalation, updated_session)
#         reply, escalation, updated_session = await handle_text_message(payload.user_text, temp_session)

#         # 3. Update History for Response
#         # We append the new interaction so .NET can save it to the DB
#         new_history = payload.conversation_history + [
#             ChatMessage(role="user", content=payload.user_text),
#             ChatMessage(role="assistant", content=reply)
#         ]

#         return ChatResponse(
#             reply=reply,
#             conversation_history=new_history,
#             session_intent=updated_session.get("intent") or updated_session.get("session_intent") or "unknown"
#         )

#     except Exception as e:
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=f"Text chat processing failed: {str(e)}")


# # -----------------------
# # VOICE ENDPOINT
# # -----------------------
# @app.post("/voice/process", response_model=VoiceResponse)
# async def voice_process(payload: VoiceRequest):
#     """
#     Handles full voice turn (ASR -> Intent -> RAG -> TTS).
#     Expects JSON with base64 audio, returns JSON with transcripts and audio.
#     """
#     try:
#         # 1. Decode Audio
#         audio_bytes = decode_audio(payload.audio_b64)
#         client = get_azure_services()

#         # 2. Process Audio (ASR & Logic)
#         # Note: You need to ensure 'process_user_utterance' or a new function can handle raw bytes 
#         # and return the text/intent/reply. 
#         # Since we can't see 'call.py', this block simulates the orchestration required.
        
#         # --- LOGIC START ---
#         # TODO: Replace this simulation with actual calls to your modules
#         # example: transcription = speech_service.transcribe(audio_bytes)
        
#         # For now, we reuse the existing pattern where we might need to capture outputs
#         # This is a placeholder for the actual logic flow:
#         transcript = "Transcribed text placeholder" 
#         intent = "general_inquiry"
#         reply_text = "This is a processed voice reply."
#         tts_audio = b"" # filled with actual TTS bytes if generated
#         # --- LOGIC END ---

#         # 3. Return Response matching .NET 'AIVoiceApiResponse'
#         return VoiceResponse(
#             text=transcript,
#             intent=intent,
#             reply=reply_text,
#             audio_b64=encode_audio(tts_audio) if tts_audio else None,
#             content_type="audio/wav"
#         )

#     except Exception as e:
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=f"Voice processing failed: {str(e)}")

# # -----------------------
# # LIVE CALL WS (Legacy/Mobile direct)
# # -----------------------
# @app.websocket("/ws/livecall")
# async def ws_livecall(ws: WebSocket):
#     await ws.accept()
#     client = get_azure_services()
#     async def get_audio_chunk():
#         try:
#             while True:
#                 data = await ws.receive_bytes()
#                 yield data
#         except WebSocketDisconnect:
#             return
#     async def send_audio_chunk(audio_bytes: bytes):
#         await ws.send_bytes(audio_bytes)
#     try:
#         await handle_live_call(get_audio_chunk, send_audio_chunk, client)
#     except WebSocketDisconnect:
#         print("Client disconnected")
#     except Exception:
#         traceback.print_exc()

# # -----------------------
# # HEALTH
# # -----------------------
# @app.get("/health")
# def health():
#     return {"status": "ok"}

# # In your app.py (FastAPI endpoint)
# @app.post("/api/generate-avatar-response")
# async def generate_avatar_response(query_data: dict):
#     # 1. Your existing pipeline
#     user_text = await transcribe_audio(query_data.get("audio"))
#     llm_response = await generate_llm_response(user_text)
    
#     # 2. Send to D-ID
#     d_id = DIDManager()
#     talk_result = d_id.create_talk(
#         script_text=llm_response,
#         avatar_id="your_did_avatar_id",
#         voice_id="en-US-JennyNeural"
#     )
    
#     # 3. Return the direct WebRTC URL to frontend
#     return {
#         "user_text": user_text,
#         "llm_response": llm_response,
#         "avatar_stream_url": talk_result["web_url"]  # Frontend uses this
#     }


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)