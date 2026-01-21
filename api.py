import base64
import traceback
import json
from typing import Optional, List
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect, APIRouter, Form
from pydantic import BaseModel
import uvicorn
from text_handler import handle_text_message
from voice import handle_live_call
from voice_agent import voice_cso 



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
    audio_b64: Optional[str] = None
    content_type: str = "audio/wav"
    conversation_history: List[ChatMessage] = []
    session_intent: str = ""


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
async def process_voice_file(
    file: UploadFile = File(...),
    session_intent: str = Form(""),
    conversation_history: str = Form("[]")  # JSON string
):
    """
    Accepts WAV audio with conversation context → runs Voice CSO → returns transcript, intent, reply & audio
    """
    try:
        # 1. Parse conversation history from JSON
        history_data = json.loads(conversation_history)
        history = [ChatMessage(**msg) for msg in history_data]
        
        # 2. Validate file
        if not file.filename.lower().endswith(".wav"):
            raise HTTPException(status_code=400, detail="Only WAV audio is supported")

        audio_bytes = await file.read()

        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")

        # 3. Prepare session state (same as chat endpoint)
        session_state = {
            "conversation_history": [
                {"role": msg.role, "content": msg.content}
                for msg in history
            ],
            "session_intent": session_intent,
        }

        # 4. Process voice - voice_cso needs to be modified to accept session_state
        # The voice_cso function should return a dict with conversation_history and session_intent
        result = await voice_cso(audio_bytes, session_state)

        # 5. Return response with updated context
        new_history = [
            ChatMessage(role=m["role"], content=m["content"])
            for m in result.get("conversation_history", session_state["conversation_history"])
        ]

        return VoiceResponse(
            text=result.get("transcript", ""),
            intent=result.get("intent", ""),
            reply=result.get("reply", ""),
            audio_b64=base64.b64encode(result["audio"]).decode() if result.get("audio") else None,
            conversation_history=new_history,
            session_intent=result.get("session_intent", session_intent)
        )

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid conversation_history JSON")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

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

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)